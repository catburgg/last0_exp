import os
import json
import torch
import logging
import argparse
import random
import shutil
from typing import List, Dict, Any
from scipy.spatial.transform import Rotation as R
from PIL import Image
import wandb
from tqdm import tqdm
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from transformers import (
    set_seed,
)

from janus.models import VLChatProcessor, ActionTokenizer
from transformers import AutoModelForCausalLM
import PIL.Image

from torch.optim.lr_scheduler import LambdaLR
import math
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

from dataclasses import dataclass
@dataclass
class VLChatProcessorOutput():
    sft_format: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    num_image_tokens: torch.IntTensor

    def __len__(self):
        return len(self.input_ids)


def get_custom_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    min_lr_ratio=0.0, 
    num_cycles=0.5
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
        scaled_factor = (1 - min_lr_ratio) * cosine_factor + min_lr_ratio
        return scaled_factor

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

def get_learning_rate(step, initial_lr, num_warmup_steps, num_training_steps, min_lr_ratio, num_cycles=0.5):
    if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps)) * initial_lr
    progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
    scaled_factor = (1 - min_lr_ratio) * cosine_factor + min_lr_ratio
    return scaled_factor * initial_lr


class TrainingMetrics:
    def __init__(self, device):
        self.n_step = 0
        self.image_right = torch.Tensor([0]).to(device=device)
        self.image_total = torch.Tensor([0]).to(device=device)
        self.image_loss = torch.Tensor([0]).to(device=device)
        self.world_size = dist.get_world_size()

    def __call__(self, image_logits, image_labels, image_loss):
        return self.update(image_logits, image_labels, image_loss)

    def update(self, image_logits, image_labels, image_loss):
        self.n_step += 1
        with torch.no_grad():
            shift_image_preds = image_logits.argmax(dim=-1) # logits[..., :-1, :].argmax(dim=-1)
            shift_image_labels = image_labels # labels[..., 1:]
            self.image_right += (shift_image_preds == shift_image_labels).masked_fill(shift_image_labels.eq(-100), 0).sum().item()
            self.image_total += (shift_image_labels != -100).sum().item()
            self.image_loss += image_loss.item()

    def get_metric(self, reset=True):
        dist.all_reduce(self.image_right, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.image_total, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.image_loss, op=torch.distributed.ReduceOp.SUM)

        image_acc = (self.image_right / self.image_total).item()
        image_loss = self.image_loss.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            self.image_right.fill_(0)
            self.image_total.fill_(0)
            self.image_loss.fill_(0)
        return image_acc, image_loss


class SftDataset(Dataset):
    def __init__(self, config, processor,accelerator, model):
        self.config = config
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.action_tokenizer = ActionTokenizer(self.tokenizer)
        self.accelerator = accelerator
        self.image_len = 576
        with open(config.data_path,'r') as f:
            self.data = json.load(f)
        accelerator.print(f'Total data amount: {len(self.data)}')

  
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.data[index]

    def crop_by_saturation(input_path, pad=8):
        img = Image.open(input_path).convert("RGB")
        W, H = img.size
        hsv = np.array(img.convert("HSV"))
        s = hsv[:, :, 1].astype(np.float32) / 255.0
        v = hsv[:, :, 2].astype(np.float32) / 255.0
        mask = ((s > 0.15) | (v < 0.4)).astype(np.uint8)

        rows = np.where(mask.sum(axis=1) > 0)[0]
        cols = np.where(mask.sum(axis=0) > 0)[0]
        if rows.size and cols.size:
            y0, y1 = int(rows[0]), int(rows[-1] + 1)
            x0, x1 = int(cols[0]), int(cols[-1] + 1)
        else:
            x0, y0, x1, y1 = 0, 0, W, H

        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(W, x1 + pad)
        y1 = min(H, y1 + pad)

        cropped = img.crop((x0, y0, x1, y1))
        # cropped.save(out_cropped)

        # # 保存掩码图（便于调参）
        # mask_img = Image.fromarray((mask * 255).astype('uint8'))
        # mask_img.save(out_mask)

        # info = {
        #     "input": input_path,
        #     "out_cropped": out_cropped,
        #     "out_mask": out_mask,
        #     "s_threshold": s_thresh,
        #     "pad": pad,
        #     "crop_box": (x0, y0, x1, y1),
        #     "crop_size": (x1 - x0, y1 - y0),
        #     "orig_size": (W, H),
        # }
        return info

    def process_image(self,image_paths):
        # images = [PIL.Image.open(image_path).convert("RGB").resize((384, 384), Image.BICUBIC) for image_path in image_paths]
        images = [PIL.Image.open(image_path).convert("RGB").resize((384, 384), Image.BICUBIC) for image_path in image_paths]
        images_outputs = self.processor.image_processor(images, return_tensors="pt")
        return images_outputs['pixel_values']

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        gen_images = [x['output_image'] for x in batch]
        input_images = sum([x['input_image'] for x in batch if 'input_image' in x],[])
        input_images = [x for x in input_images]

        pixel_values = self.process_image(gen_images).to(torch.bfloat16)
        input_pixel_values = self.process_image(input_images).to(torch.bfloat16) if len(input_images) > 0 else None

        input_img_tokens = self.processor.image_start_tag + self.processor.pad_tag*self.processor.num_image_tokens +self.processor.image_end_tag
        output_img_tokens = self.processor.image_start_tag + self.processor.pad_tag*self.processor.num_image_tokens if self.config.image_generation else ""

        pre_data = []

        for x in batch:
            img_len = len(x['input_image']) if 'input_image' in x and len(x['input_image']) > 0 else 0
            prompts = x['input_prompt'] + input_img_tokens * img_len

            conversation = [
                {"role": "<|User|>","content": prompts},
                {"role": "<|Assistant|>", "content": ""}
            ]

            pre_format = self.processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.processor.sft_format,
                system_prompt="",
            )
            sft_format = pre_format + output_img_tokens
            
            if img_len > 0:
                encoder_pixel_values = self.process_image([input_img for input_img in x['input_image']])
                num_image_tokens = [self.image_len] * img_len
            else:
                encoder_pixel_values = None
                num_image_tokens = []
            
            input_ids =  torch.LongTensor(self.processor.tokenizer.encode(sft_format))
            pre_data.append(VLChatProcessorOutput(sft_format=sft_format, pixel_values=encoder_pixel_values, input_ids=input_ids, num_image_tokens=num_image_tokens))


        if len(pre_data) > 0:
            prepare_inputs = self.processor.batchify(pre_data)

        return {
            "input_ids": prepare_inputs.input_ids,
            "pixel_values": pixel_values,
            "input_pixel_values": input_pixel_values,
            "encoder_pixel_values": prepare_inputs.pixel_values.to(torch.bfloat16),
            "attention_mask": prepare_inputs.attention_mask,
            "images_seq_mask": prepare_inputs['images_seq_mask'],
            "images_emb_mask": prepare_inputs['images_emb_mask']
        }


def save_checkpoint(
    model,
    processor,
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch: int,
    step: int,
    global_step: int,
    is_last: bool = False,
    stats_data = None
) -> None:

    save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
    
    if accelerator.is_main_process:
        # Manage checkpoint numbers
        checkpoint_files = [f for f in os.listdir(args.output_dir) if f.startswith("checkpoint-")]
        if args.max_ckpts > 0 and len(checkpoint_files) >= args.max_ckpts:
            oldest_ckpt = min(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
            shutil.rmtree(os.path.join(args.output_dir, oldest_ckpt))

        os.makedirs(save_dir, exist_ok=True)
        output_dir = os.path.join(save_dir, 'tfmr')

        model.save_pretrained(output_dir, state_dict=accelerator.get_state_dict(model))
        processor.save_pretrained(output_dir)

    accelerator.wait_for_everyone()
    logger.info(f'Checkpoint {epoch}-{global_step} saved successfully')



def train(args: argparse.Namespace) -> None:

    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    # Set random seed
    set_seed(args.seed)

    if accelerator.is_main_process:
        wandb.init(
            project=args.experiment_name,
            name=args.run_name,
            config=args,
            dir=args.log_dir,
            mode="online"
        )

    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = (
        args.train_bsz_per_gpu * 
        dist.get_world_size() * 
        accelerator.gradient_accumulation_steps
    )

    processor = VLChatProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    model_config = model.config

    for name, param in model.named_parameters():
        if name.startswith("vision_model") or name.startswith("aligner") or name.startswith("gen_vision_model"): # choose whatever you like here
            param.requires_grad = False

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    train_dataset = SftDataset(args, processor,accelerator,model)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_bsz_per_gpu,
        shuffle=True,
        # drop_last=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=8
    )

    num_training_steps = int(len(train_dataloader) * args.n_epochs) // accelerator.gradient_accumulation_steps // dist.get_world_size()

    lr_scheduler = get_custom_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_rates * num_training_steps),
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr_ratio  # Pass minimum learning rate ratio directly
    )

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    metric = TrainingMetrics(device=torch.cuda.current_device())
    model.train()
    global_step = 0

    for epoch in range(0, args.n_epochs):
        train_iter = tqdm(train_dataloader, total=len(train_dataloader)) if accelerator.is_main_process else train_dataloader
        for batch in train_iter:

            if batch['input_pixel_values'] is not None:
                quant_input, emb_loss_input, info_input = model.gen_vision_model.encode(batch['input_pixel_values'])
                image_tokens_input = info_input[2].detach().reshape(batch['input_pixel_values'].shape[0], -1)
                # image_tokens_input = info_input[2].reshape(batch['input_pixel_values'].shape[0], -1)
                image_embeds_input = model.prepare_gen_img_embeds(image_tokens_input)

                # torch.set_printoptions(threshold=10_000)
                # print(batch['input_ids'])
                # print(batch['input_ids'].shape, batch['images_emb_mask'].shape, batch['images_seq_mask'].shape)
                # print(batch['images_emb_mask'])
                # print(batch['images_seq_mask'])
                # import time
                # time.sleep(30)

                batch['input_ids'][batch['input_ids'] < 0] = 0  # ignore the image embeddings
                inputs_embeds = model.language_model.get_input_embeddings()(batch['input_ids'])

                # Find the position of the input image gen and concatenate it
                image_gen_indices = (batch['input_ids'] == processor.image_start_id).nonzero()
                if args.image_generation:
                    image_gen_indices = [
                        ind for ii, ind in enumerate(image_gen_indices) 
                        if (ii + 1) % (image_embeds_input.shape[0] // args.train_bsz_per_gpu + 1) != 0
                    ]
                for in_img_index, ind in enumerate(image_gen_indices):
                    offset = ind[1] + 1
                    inputs_embeds[ind[0], offset:offset+image_embeds_input.shape[1], :] = image_embeds_input[in_img_index]
            else:
                inputs_embeds = model.prepare_inputs_embeds(
                    input_ids=batch['input_ids'],
                    pixel_values=batch['encoder_pixel_values'],
                    images_emb_mask=batch['images_emb_mask'],
                    images_seq_mask=batch['images_seq_mask']
                )
            
            if args.image_generation:
                quant, emb_loss, info = model.gen_vision_model.encode(batch['pixel_values'])
                image_tokens = info[2].detach().reshape(batch['pixel_values'].shape[0], -1).contiguous()
                # image_tokens = info[2].reshape(batch['pixel_values'].shape[0], -1).contiguous()
                image_embeds = model.prepare_gen_img_embeds(image_tokens)
                inputs_embeds[:, -image_embeds.shape[1]:,:] = image_embeds

            outputs = model.language_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=batch['attention_mask'],
                return_dict=True,
                use_cache=False
            )
            
            hidden_states = outputs.last_hidden_state

            if args.image_generation:
                ### +2 means: <image start token> and <prompt end token>
                # action_tokens = batch['input_ids'][:, -(image_embeds.shape[1]+2+args.action_dim) : -(image_embeds.shape[1]+2)].contiguous()
                # action_logits = model.language_model.lm_head(hidden_states[:, -(image_embeds.shape[1]+3+args.action_dim) : -(image_embeds.shape[1]+3), :])
                # action_loss = model.language_model.loss_function(logits=action_logits, labels=None, vocab_size=action_logits.shape[-1], shift_labels=action_tokens)
                image_logits = model.gen_head(hidden_states[:, -(image_embeds.shape[1]+1) : -1, :])
                image_loss = model.language_model.loss_function(logits=image_logits, labels=None, vocab_size=model_config.gen_vision_config.params.image_token_size, shift_labels=image_tokens)
                # loss = action_loss * 0.5 + image_loss * 0.5
                loss = image_loss
                metric(image_logits, image_tokens, image_loss)
            else:
                ### +1 means: <prompt end token>
                action_tokens = batch['input_ids'][:, -(args.action_dim+1) : -1].contiguous()
                action_logits = model.language_model.lm_head(hidden_states[:, -(2+args.action_dim) : -2, :])
                action_loss = model.language_model.loss_function(logits=action_logits, labels=None, vocab_size=action_logits.shape[-1], shift_labels=action_tokens)
                loss = action_loss
                metric(args.image_generation, None, None, None, action_logits, action_tokens, action_loss)

            accelerator.backward(loss)

            if (global_step + 1) % accelerator.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                iamge_acc, image_loss= metric.get_metric() if args.image_generation else metric.get_metric_action()
                if accelerator.is_main_process:
                    train_iter.set_postfix(
                        epoch=epoch,
                        step=global_step,
                        total_steps=len(train_dataloader),
                        skip=accelerator.optimizer_step_was_skipped,
                        length=len(batch["input_ids"][0]),
                        image_loss=f"{image_loss:.6f}",
                        iamge_acc=f"{iamge_acc:.6f}",
                        lr=f"{lr_scheduler.get_last_lr()[0]:.2e}"
                    )
                    wandb.log({
                        'image_loss': image_loss,
                        'iamge_acc': iamge_acc,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }, step=global_step)

            global_step += 1

        # if epoch == args.n_epochs-1:
        #     accelerator.wait_for_everyone()
        #     save_checkpoint(
        #         model=model,
        #         processor=processor, 
        #         accelerator=accelerator,
        #         args=args,
        #         epoch=epoch,
        #         step=global_step-1,
        #         global_step=global_step,
        #         is_last=True,
        #         stats_data = train_dataset.stats_data,
        #     )

        if epoch == args.n_epochs-1:
            accelerator.wait_for_everyone()
            save_checkpoint(
                model=model,
                processor=processor, 
                accelerator=accelerator,
                args=args,
                epoch=epoch,
                step=global_step-1,
                global_step=global_step,
                is_last=(epoch == args.n_epochs-1),
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-training parameter configuration')
    
    # Experiment settings
    parser.add_argument('--experiment_name', type=str, default='janus_train', help='Experiment name')
    parser.add_argument('--run_name', type=str, default='run_1', help='Run name')
    parser.add_argument('--model_path', type=str, default='', help='Pre-trained model path')

    # Data related
    parser.add_argument('--data_path', type=str, required=True, help='Training data path, can be multiple paths')
    parser.add_argument('--data_root', type=str, required=True, default='')
    parser.add_argument('--output_dir', type=str, default='./', help='Model save path')
    parser.add_argument('--max_ckpts', type=int, default=5, help='Maximum number of checkpoints to save')
    parser.add_argument('--log_dir', type=str, default='./train_logs', help='Log save path')
    parser.add_argument('--action_dim', type=int, default=7, help='action dim')
    parser.add_argument('--robot_state', action='store_true', default=False, help='enable robot state')
    parser.add_argument('--image_generation', type=int, default=0, help='generate image')

    # Training related
    parser.add_argument('--max_seq_len', type=int, default=4096, help='Maximum sequence length')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping threshold, set to 0 for no clipping')
    parser.add_argument('--train_bsz_per_gpu', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--min_lr_ratio', type=float, default=0., help='Minimum learning rate ratio to peak learning rate')
    parser.add_argument('--warmup_rates', type=float, default=0., help='Warmup ratio')
    parser.add_argument('--n_epochs', type=int, default=3, help='Number of training epochs')

    # Others
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    
    # Set paths
    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    if args.run_name:
        args.output_dir = os.path.join(args.output_dir, args.run_name)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Start training
    train(args)     