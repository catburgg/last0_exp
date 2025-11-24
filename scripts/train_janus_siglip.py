import os
import json
import torch
import logging
import argparse
import random
import shutil
import math
import wandb
import PIL.Image
import numpy as np
import time

from typing import List, Dict, Any
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator
from einops import rearrange
from transformers import (
    set_seed,
)
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor, ActionTokenizer


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


def create_component_indexes(seq_len, action_len=7):
    latent_indexes = torch.arange(0, seq_len - action_len)
    action_indexes = torch.arange(seq_len - action_len, seq_len)
    return latent_indexes, action_indexes


class TrainingMetrics:
    def __init__(self, device):
        self.n_step = 0
        self.action_right = torch.Tensor([0]).to(device=device)
        self.action_total = torch.Tensor([0]).to(device=device)
        self.action_loss = torch.Tensor([0]).to(device=device)
        self.sim_loss = torch.Tensor([0]).to(device=device)
        self.world_size = dist.get_world_size()

    def __call__(self, has_latent, action_logits, action_labels, action_loss, sim_loss):
        if has_latent:
            return self.update(action_logits, action_labels, action_loss, sim_loss)
        else:
            return self.update_action(action_logits, action_labels, action_loss, sim_loss)

    def update(self, action_logits, action_labels, action_loss, sim_loss):
        self.n_step += 1
        with torch.no_grad():
            shift_action_preds = action_logits.argmax(dim=-1) # logits[..., :-1, :].argmax(dim=-1)
            shift_action_labels = action_labels # labels[..., 1:]
            # print("label", shift_action_labels[0])
            # print("pred", shift_action_preds[0])
            # print()
            self.action_right += (shift_action_preds == shift_action_labels).masked_fill(shift_action_labels.eq(-100), 0).sum().item()
            self.action_total += (shift_action_labels != -100).sum().item()
            self.action_loss += action_loss.item()
            self.sim_loss += sim_loss.item()

    def update_action(self, action_logits, action_labels, action_loss, sim_loss):
        self.n_step += 1
        with torch.no_grad():
            shift_action_preds = action_logits.argmax(dim=-1) # logits[..., :-1, :].argmax(dim=-1)
            shift_action_labels = action_labels # labels[..., 1:]
            self.action_right += (shift_action_preds == shift_action_labels).masked_fill(shift_action_labels.eq(-100), 0).sum().item()
            self.action_total += (shift_action_labels != -100).sum().item()
            self.action_loss += action_loss.item()
            self.sim_loss += sim_loss.item()

    def get_metric(self, reset=True):
        dist.all_reduce(self.action_right, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.action_total, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.action_loss, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.sim_loss, op=torch.distributed.ReduceOp.SUM)

        action_acc = (self.action_right / self.action_total).item()
        action_loss = self.action_loss.item() / (self.world_size * self.n_step)
        sim_loss = self.sim_loss.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            self.action_right.fill_(0)
            self.action_total.fill_(0)
            self.action_loss.fill_(0)
            self.sim_loss.fill_(0)
        return action_acc, action_loss, sim_loss

    def get_metric_action(self, reset=True):
        dist.all_reduce(self.action_right, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.action_total, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.action_loss, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.sim_loss, op=torch.distributed.ReduceOp.SUM)
        action_acc = (self.action_right / self.action_total).item()
        action_loss = self.action_loss.item() / (self.world_size * self.n_step)
        sim_loss = self.sim_loss.item() / (self.world_size * self.n_step)   

        if reset:
            self.n_step = 0
            self.action_right.fill_(0)
            self.action_total.fill_(0)
            self.action_loss.fill_(0)
            self.sim_loss.fill_(0)
        return 0, 0, action_acc, action_loss, sim_loss


class SftDataset(Dataset):
    def __init__(self, config, processor,accelerator, model):
        self.config = config
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.action_tokenizer = ActionTokenizer(self.tokenizer, need_to_sub=3) # 3 for latent spetial tokens
        self.accelerator = accelerator
        self.image_len = 576
        with open(config.data_path,'r') as f:
            self.data = json.load(f)

        statistics_path = config.data_path.replace(".json", "_statistics.json")
        with open(statistics_path, 'r') as f:
            self.stats_data = json.load(f)

        self.dataset_name = next(iter(self.stats_data))
        self.action_mask = np.array(self.stats_data[self.dataset_name]['action']['mask'])
        self.action_min = np.array(self.stats_data[self.dataset_name]['action']['q01'])
        self.action_max = np.array(self.stats_data[self.dataset_name]['action']['q99'])
        self.state_mask = np.array(self.stats_data[self.dataset_name]['state']['mask'])
        self.state_min = np.array(self.stats_data[self.dataset_name]['state']['q01'])
        self.state_max = np.array(self.stats_data[self.dataset_name]['state']['q99'])

        self.img_dir = os.path.dirname(config.data_path)
        accelerator.print(f'Total data amount: {len(self.data)}')

  
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.data[index]

    def process_image(self,image_paths):
        images = [PIL.Image.open(image_path).convert("RGB") for image_path in image_paths]
        images_outputs = self.processor.image_processor(images, return_tensors="pt")
        return images_outputs['pixel_values']

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_images = sum([x['input_image'] for x in batch if 'input_image' in x],[])
        input_images = [os.path.join(self.img_dir,x) for x in input_images]

        if self.config.use_latent:
            latent_images = [os.path.join(self.img_dir, x['output_image']) for x in batch]
            latent_pixel_values = self.process_image(latent_images).to(torch.bfloat16) if len(latent_images) > 0 else None
        else:
            latent_pixel_values = None

        input_img_tokens = self.processor.image_start_tag + self.processor.image_tag*self.processor.num_image_tokens +self.processor.image_end_tag

        latent_start_str = "<|latent_start|>"
        latent_pad_str = "<|latent_pad|>" * self.config.latent_size
        latent_end_str = "<|latent_end|>"

        pre_data = []

        for x in batch:
            img_len = len(x['input_image']) if 'input_image' in x and len(x['input_image']) > 0 else 0

            action = np.array(x['action'], dtype=np.float32)
            normalized_action = np.where(
                self.action_mask,
                np.clip(2 * (action - self.action_min) / (self.action_max - self.action_min + 1e-8) - 1, -1, 1),
                action
            )
            action_tokens = ""
            action_tokens += self.action_tokenizer(normalized_action)

            state_tokens = ""
            if self.config.robot_state:
                state = np.array(x['state'], dtype=np.float32)
                normalized_state = np.where(
                    self.state_mask,
                    np.clip(2 * (state - self.state_min) / (self.state_max - self.state_min + 1e-8) - 1, -1, 1),
                    state
                )
                state_tokens += self.action_tokenizer(normalized_state)

            latent_str = latent_start_str + latent_pad_str + latent_end_str if self.config.use_latent else ""
            prompts = input_img_tokens * img_len + x['input_prompt'] + state_tokens + latent_str + action_tokens

            conversation = [
                {"role": "<|User|>","content": prompts},
            ]

            pre_format = self.processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.processor.sft_format,
                system_prompt="",
            )
            sft_format = pre_format
            
            if img_len > 0:
                encoder_pixel_values = self.process_image([os.path.join(self.img_dir,input_img) for input_img in x['input_image']])
                num_image_tokens = [self.image_len] * img_len
            else:
                encoder_pixel_values = None
                num_image_tokens = []
            
            input_ids = torch.LongTensor(self.processor.tokenizer.encode(sft_format))
            pre_data.append(
                VLChatProcessorOutput(
                    sft_format=sft_format, 
                    pixel_values=encoder_pixel_values, 
                    input_ids=input_ids, 
                    num_image_tokens=num_image_tokens
                )
            )

        if len(pre_data) > 0:
            prepare_inputs = self.processor.batchify(pre_data)

        return {
            "input_ids": prepare_inputs.input_ids,
            "encoder_pixel_values": prepare_inputs.pixel_values.to(torch.bfloat16),
            "latent_pixel_values": latent_pixel_values,
            "attention_mask": prepare_inputs.attention_mask,
            "images_seq_mask": prepare_inputs['images_seq_mask'],
            "images_emb_mask": prepare_inputs['images_emb_mask'],
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

        with open(os.path.join(save_dir, 'stats_data.json'), 'w') as f:
            json.dump(stats_data, f, indent=2)
            
        logger.info(f"Statistics have been saved to {os.path.join(save_dir, 'stats_data.json')}")

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
        if '_action' in name:
            if args.load_action_from_latent and name.endswith('.weight'):
                if 'embed_tokens' in name: # actually not used
                    base_name = name.replace('_action', '')
                    base_embed_weight = model.state_dict()[base_name]
                    last_259_tokens = base_embed_weight[-259:] # 259: 256 action tokens + 3 latent tokens
                    param.data.copy_(last_259_tokens)
                    accelerator.print(f"Initialized {name} with last 259 tokens from embed_tokens")
                elif 'lm_head' in name: # actually not used
                    base_name = name.replace('_action', '')
                    base_embed_weight = model.state_dict()[base_name]
                    last_259_tokens = base_embed_weight[-259:]
                    param.data.copy_(last_259_tokens)
                    accelerator.print(f"Initialized {name} with last 259 tokens from embed_tokens")
                else:
                    base_name = name.replace('_action', '')
                    if base_name in model.state_dict():
                        param.data.copy_(model.state_dict()[base_name])
                        accelerator.print(f"Initialized {name} from {base_name}")
            param.requires_grad = True
        else:
            if args.freeze_latent:
                if 'lm_head' in name or 'embed_tokens' in name: # important!: action also need lm_head to decode
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                if any(name.startswith(prefix) for prefix in ["vision_model", "aligner", "gen_vision_model"]):
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    accelerator.print(f"Freeze latent: {args.freeze_latent}")
    accelerator.print(f"Load action from latent: {args.load_action_from_latent}")
    accelerator.print(f"Total parameters: {total_params/1e9:.2f}B")
    accelerator.print(f"Trainable parameters: {trainable_params/1e9:.2f}B")
    accelerator.print(f"Non-trainable parameters: {non_trainable_params/1e9:.2f}B")
    accelerator.print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    if args.freeze_latent:
        accelerator.print("Freeze strategy: Training only parameters with '_action' in name")
    else:
        accelerator.print("Freeze strategy: Freezing only vision-related parameters (vision_model, aligner, gen_vision_model)")

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    train_dataset = SftDataset(args, processor, accelerator, model)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_bsz_per_gpu,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=8
    )

    num_training_steps = int(len(train_dataloader) * args.n_epochs) // accelerator.gradient_accumulation_steps // dist.get_world_size()
    lr_scheduler = get_custom_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_rates * num_training_steps),
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr_ratio
    )
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    metric = TrainingMetrics(device=torch.cuda.current_device())
    model.train()
    global_step = 0

    for epoch in range(0, args.n_epochs):

        train_iter = tqdm(train_dataloader, total=len(train_dataloader)) if accelerator.is_main_process else train_dataloader
        for batch in train_iter:
            inputs_embeds = model.prepare_inputs_embeds(
                    input_ids=batch['input_ids'],
                    pixel_values=batch['encoder_pixel_values'],
                    images_emb_mask=batch['images_emb_mask'],
                    images_seq_mask=batch['images_seq_mask']
                )
            latent_indexes, action_indexes = create_component_indexes(inputs_embeds.shape[1], args.action_dim+1) # important: +1 for <latent_end>
            # print(batch['input_ids'][0], batch['input_ids'].shape)
            # print(latent_indexes, action_indexes)
            # input("check indexes")
            
            if args.use_latent:
                
                batch['latent_pixel_values'] = batch['latent_pixel_values'].unsqueeze(1)
                bs, n = batch['latent_pixel_values'].shape[0:2]  # [B, n_images, 3, H, W]
                helper_images = rearrange(batch['latent_pixel_values'], "b n c h w -> (b n) c h w")
                latent_embeds = model.aligner(model.vision_model(helper_images))
                latent_embeds = rearrange(latent_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
                latent_embeds_full = latent_embeds

                B, T_full, D = latent_embeds_full.shape

                # calculate the latent embeddings (GT) for loss computation
                latent_size = args.latent_size
                compress_strategy = args.compress_strategy
                compressed_latent_embeds = []
                for b in range(B):
                    feats = latent_embeds_full[b, :, :]
                    if feats.shape[0] != latent_size:
                        if compress_strategy == "average":
                            group_size = max(1, feats.shape[0] // latent_size)
                            res = feats.shape[0] % latent_size
                            if res > 0:
                                feats = feats[:-res, :]
                            chunks = torch.split(feats, group_size, dim=0)
                            feats = torch.cat([c.mean(dim=0, keepdim=True) for c in chunks], dim=0)
                        else:
                            feats = feats[:latent_size, :]
                    compressed_latent_embeds.append(feats)
                # latent embeddings (GT)
                compressed_latent_embeds = torch.stack(compressed_latent_embeds, dim=0).to(inputs_embeds.dtype)  # [B, latent_size, D]

                # ------lzy: latent cot progress, not sure whether to use------
                # latent chain-of-thought process
                input_ids_for_cot = batch['input_ids'].clone()
                latent_indices = (input_ids_for_cot == 100847).nonzero()
                # print("input_ids_for_cot[0]", input_ids_for_cot[0])
                # print("input_ids_for_cot.shape", input_ids_for_cot.shape)
                # print("latent indices", latent_indices)
                # input()

                latent_lists = [[idx[1].item() for idx in latent_indices if idx[0] == i] for i in range(input_ids_for_cot.shape[0])]
                kv_cache_cot = None
                next_compute_range = (0, latent_indices[:, 1].min().item()) # init the next compute range

                for latent_i in range(args.latent_size):
                    # print("next compute range: ", next_compute_range)
                    curr_input_embeds_cot = inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :]
                    outputs = model.language_model.model(
                        inputs_embeds=curr_input_embeds_cot,
                        return_dict=True,
                        use_cache=True,
                        past_key_values=kv_cache_cot if latent_i!=0 else None,
                        latent_indexes=torch.arange(0, curr_input_embeds_cot.shape[1]).to(curr_input_embeds_cot.device),
                        action_indexes=torch.arange(0, 0).to(curr_input_embeds_cot.device),
                    )

                    # update next compute range
                    next_compute_range = (
                        next_compute_range[1],
                        (
                            input_ids_for_cot.shape[1]
                            if latent_i + 1 == args.latent_size
                            else next_compute_range[1] + 1
                        )
                    )

                    hidden_states_cot = outputs[0][:, -1:, :]
                    kv_cache_cot = outputs.past_key_values
                    
                    filling_indices = [
                        (instance_idx, mask_list[latent_i])
                        for instance_idx, mask_list in enumerate(latent_lists)
                        if len(mask_list) > latent_i
                    ]
                    # print(filling_indices)
                    # input("Press Enter to continue...")

                    # break the original input embeddings into tensor list
                    tensor_list = [
                        [
                            inputs_embeds[batch_idx, pos, :]
                            for pos in range(inputs_embeds.shape[1])
                        ]
                        for batch_idx in range(inputs_embeds.shape[0])
                    ]
                    for idx_pair in filling_indices:
                        batch_idx, token_idx = idx_pair
                        tensor_list[batch_idx][token_idx] = hidden_states_cot[batch_idx][0]

                    # re-combine the tensors to input embeddings
                    inputs_embeds = torch.stack([
                        torch.stack(tensor_list[batch_idx])
                        for batch_idx in range(inputs_embeds.shape[0])
                    ])
                
                # compute the cos similarity loss of latent embeddings
                latent_pad_indices_cot = (batch['input_ids'] == 100847).nonzero()
                cot_embeds_list = []
                for batch_idx in range(inputs_embeds.shape[0]):
                    batch_latent_indices = latent_pad_indices_cot[latent_pad_indices_cot[:,0] == batch_idx]
                    if len(batch_latent_indices) > 0:
                        offset = batch_latent_indices[0,1].item()
                        # print(offset)
                        # input()
                        cot_inferred_embeddings = inputs_embeds[batch_idx, offset: offset+args.latent_size, :]
                        cot_embeds_list.append(cot_inferred_embeddings)
                inferred_embeddings_all = torch.stack(cot_embeds_list, dim=0).to(compressed_latent_embeds.dtype)
                similarity = F.cosine_similarity(inferred_embeddings_all, compressed_latent_embeds, dim=-1).mean()
                sim_loss = 1.0 - similarity
                # ------lzy: latent cot progress end------

                # ------lzy: 1102 test 1: directly use latent embedding GT as the condition, do not delete------
                # for b in range(B):
                #     mask_pos = (latent_mask[b] == 1).nonzero(as_tuple=True)[0]
                #     if len(mask_pos) > 0:
                #         replace_len = min(len(mask_pos), compressed_latent_embeds.shape[1])
                #         inputs_embeds[b, mask_pos[:replace_len], :] = compressed_latent_embeds[b, :replace_len, :]

                # forward for action prediction
                outputs = model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=batch['attention_mask'],
                    return_dict=True,
                    use_cache=False,
                    latent_indexes=latent_indexes.to(inputs_embeds.device),
                    action_indexes=action_indexes.to(inputs_embeds.device),
                    use_latent=args.use_latent,
                )
                hidden_states = outputs.last_hidden_state

                # calculate multimodal loss
                # action loss (cross entropy)
                action_tokens = batch['input_ids'][:, -args.action_dim:].contiguous()
                action_logits = model.language_model.lm_head(hidden_states[:, -(1+args.action_dim) : -1, :])
                action_loss = model.language_model.loss_function(logits=action_logits, labels=None, vocab_size=action_logits.shape[-1], shift_labels=action_tokens)
                if args.freeze_latent: # stage2
                    loss = action_loss
                else: # stage0
                    loss = sim_loss
                metric(args.use_latent, action_logits, action_tokens, action_loss, sim_loss)
            else:
                cache_position_condition = torch.arange(0, inputs_embeds.shape[1]-args.action_dim-1, device=inputs_embeds.device, dtype=torch.long)
                cache_position_action = torch.arange(inputs_embeds.shape[1]-args.action_dim+args.latent_size, inputs_embeds.shape[1]+args.latent_size+1, device=inputs_embeds.device, dtype=torch.long)
                outputs = model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=batch['attention_mask'],
                    return_dict=True,
                    use_cache=False,
                    latent_indexes=latent_indexes,
                    action_indexes=action_indexes,
                    cache_position=torch.cat([cache_position_condition, cache_position_action]),
                    use_latent=args.use_latent,
                )
                hidden_states = outputs.last_hidden_state
                
                action_tokens = batch['input_ids'][:, -args.action_dim:].contiguous()
                action_logits = model.language_model.lm_head(hidden_states[:, -(1+args.action_dim) : -1, :])
                action_loss = model.language_model.loss_function(logits=action_logits, labels=None, vocab_size=action_logits.shape[-1], shift_labels=action_tokens)
                loss = action_loss
                metric(args.use_latent, action_logits, action_tokens, action_loss)

            accelerator.backward(loss)
            if (global_step + 1) % accelerator.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                action_acc, action_loss, sim_loss= metric.get_metric() if args.use_latent else metric.get_metric_action()
                if accelerator.is_main_process:
                    train_iter.set_postfix(
                        epoch=epoch,
                        step=global_step,
                        total_steps=len(train_dataloader),
                        skip=accelerator.optimizer_step_was_skipped,
                        length=len(batch["input_ids"][0]),
                        action_loss=f"{action_loss:.6f}",
                        action_acc=f"{action_acc:.6f}",
                        sim_loss=f"{sim_loss:.6f}",
                        lr=f"{lr_scheduler.get_last_lr()[0]:.2e}"
                    )
                    wandb.log({
                        'action_loss': action_loss,
                        'action_acc': action_acc,
                        'sim_loss': sim_loss,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }, step=global_step)
            global_step += 1

        if ((epoch + 1) % 20 == 0) or (epoch == args.n_epochs-1):
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
                stats_data=train_dataset.stats_data,
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
    parser.add_argument('--max_ckpts', type=int, default=10, help='Maximum number of checkpoints to save')
    parser.add_argument('--log_dir', type=str, default='./train_logs', help='Log save path')

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
    parser.add_argument('--action_dim', type=int, default=7, help='action dim')
    parser.add_argument('--robot_state', action='store_true', default=False, help='enable robot state')
    parser.add_argument('--load_action_from_latent', type=int, default=0)
    parser.add_argument('--freeze_latent', type=int, default=0)
    parser.add_argument('--image_token_num', type=int, default=576)
    parser.add_argument('--use_latent', type=int, default=1)
    parser.add_argument('--latent_size', type=int, default=4)
    parser.add_argument('--compress_strategy',type=str, required=True,default='average')

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

