import os, sys, pathlib
import argparse
import tqdm
import shutil
import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor, ActionTokenizer
import numpy as np
import os
import PIL.Image
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torchvision
import json
import argparse
import copy
import random
from typing import List, Dict

from termcolor import cprint, colored

from lift3d.envs.rlbench_env import RLBenchEnv, RLBenchActionMode, RLBenchObservationConfig
from lift3d.helpers.gymnasium import VideoWrapper
from lift3d.helpers.common import Logger
from lift3d.helpers.graphics import EEpose
import logging
import time
from datetime import datetime

import numpy as np
import pickle

import torch
from dataclasses import dataclass
from PIL import Image

from scipy.spatial.transform import Rotation as R

@dataclass
class VLChatProcessorOutput():
    sft_format: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    num_image_tokens: torch.IntTensor

    def __len__(self):
        return len(self.input_ids)

def setup_logger(log_dir):
    log_filename = os.path.join(log_dir, "output.log")
    
    logger = logging.getLogger("RLBenchLogger")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def recreate_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path, exist_ok=True)

def model_load(args):
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    action_tokenizer = ActionTokenizer(tokenizer)

    statistics_path = os.path.join(os.path.dirname(args.model_path), "stats_data.json")
    with open(statistics_path, 'r') as f:
        stats_data = json.load(f)
    dataset_name=args.dataset_name

    statistic= {}
    statistic['action_mask'] = np.array(stats_data[dataset_name]['action']['mask'])
    statistic['action_min'] = np.array(stats_data[dataset_name]['action']['q01'])
    statistic['action_max'] = np.array(stats_data[dataset_name]['action']['q99'])
    if args.use_robot_state:
        statistic['state_mask'] = np.array(stats_data[dataset_name]['state']['mask'])
        statistic['state_min'] = np.array(stats_data[dataset_name]['state']['q01'])
        statistic['state_max'] = np.array(stats_data[dataset_name]['state']['q99'])

    return vl_gpt, vl_chat_processor, action_tokenizer, statistic

def model_predict(args, vl_gpt, vl_chat_processor, action_tokenizer, statistic, task_description, image, state, pointcloud, pre_image_dir, step):
    device = f'cuda:{args.cuda}'
    vl_gpt = vl_gpt.to(device).eval()
    parallel_size=1
    img_len = 1
    temperature = 1.0
    image_token_num_per_image = 576
    action_token_num = 7
    img_size = 384
    patch_size = 16

    state_tokens = ""
    if args.use_robot_state:
        state = np.array(state, dtype=np.float32)
        normalized_state = np.where(
            statistic['state_mask'],
            np.clip(2 * (state - statistic['state_min']) / (statistic['state_max'] - statistic['state_min'] + 1e-8) - 1, -1, 1),
            state
        )
        state_tokens += action_tokenizer(normalized_state)


    input_img_tokens_2 = vl_chat_processor.image_start_tag + vl_chat_processor.pad_tag*vl_chat_processor.num_image_tokens +vl_chat_processor.image_end_tag
    user_content = input_img_tokens_2 * img_len + task_description + state_tokens + vl_chat_processor.image_start_tag

    conversation = [
                    {"role": "<|User|>","content": user_content},
                    # {"role": "<|Assistant|>", "content": ""}
                ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )

    with torch.inference_mode():
        input_image_pixel_values = vl_chat_processor.image_processor(image, return_tensors="pt")['pixel_values'].to(torch.bfloat16).to(device)
        quant_input, emb_loss_input, info_input = vl_gpt.gen_vision_model.encode(input_image_pixel_values)
        image_tokens_input = info_input[2].detach().reshape(input_image_pixel_values.shape[0], -1)
        image_embeds_input = vl_gpt.prepare_gen_img_embeds(image_tokens_input)

        input_ids =  torch.LongTensor(vl_chat_processor.tokenizer.encode(sft_format))
        tokens = torch.zeros((parallel_size, len(input_ids)), dtype=torch.long).to(device)

        for i in range(parallel_size):
            tokens[i, :] = input_ids
        image_embeds_input = image_embeds_input.repeat(parallel_size, 1, 1)

        # torch.set_printoptions(threshold=10_000)
        # print(tokens)
        # print(image_embeds_input.shape)

        tokens[tokens < 0] = 0  # ignore the image embeddings
        inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
        image_gen_indices = (tokens == vl_chat_processor.image_start_id).nonzero()
        if args.image_generation:
            image_gen_indices = [
                ind for ii, ind in enumerate(image_gen_indices) 
                if (ii + 1) % 2 != 0
            ]
        for in_img_index, ind in enumerate(image_gen_indices):
            offset = ind[1] + 1
            inputs_embeds[ind[0], offset:offset+image_embeds_input.shape[1], :] = image_embeds_input[in_img_index]

        # --------------generate image------------ #
        # if args.image_generation:
        #     generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)
        #     img_embeds_list = []
        #     for i in range(image_token_num_per_image):
        #         if i == 0:
        #             outputs = vl_gpt.language_model.model(
        #                 inputs_embeds=inputs_embeds, 
        #                 use_cache=True,
        #                 image_indexes=torch.arange(0, inputs_embeds.shape[1]).to(device),
        #                 action_indexes=torch.arange(0, 0).to(device)
        #             )
        #         else:
        #             outputs = vl_gpt.language_model.model(
        #                 inputs_embeds=cur_inputs_embeds, 
        #                 use_cache=True,
        #                 past_key_values=outputs.past_key_values,
        #                 image_indexes=torch.arange(0, 1).to(device),
        #                 action_indexes=torch.arange(0, 0).to(device)
        #             )
        #         hidden_states = outputs.last_hidden_state
        #         logits = vl_gpt.gen_head(hidden_states[:, -1, :])

        #         # ch: ------ #
        #         # probs = torch.softmax(logits / temperature, dim=-1)
        #         # next_token = torch.multinomial(probs, num_samples=1)
        #         next_token = torch.argmax(logits, dim=-1, keepdim=True)
        #         # ch: ------ #

        #         generated_tokens[:, i] = next_token.squeeze(dim=-1)
        #         next_token = next_token.view(-1)
        #         img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
        #         cur_inputs_embeds = img_embeds.unsqueeze(dim=1)
        #         img_embeds_list.append(cur_inputs_embeds)

        #     dec = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
        #     dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        #     dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        #     visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        #     visual_img[:, :, :] = dec

        #     for i in range(parallel_size):
        #         save_path = os.path.join(pre_image_dir, f'step_{step}.png')
        #         PIL.Image.fromarray(visual_img[i]).save(save_path)

        if args.image_generation:
            generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)
            img_embeds_list = []
            for i in range(image_token_num_per_image):
                outputs = vl_gpt.language_model.model(
                    inputs_embeds=inputs_embeds, 
                    image_indexes=torch.arange(0, inputs_embeds.shape[1]).to(device),
                    action_indexes=torch.arange(0, 0).to(device)
                )
                hidden_states = outputs.last_hidden_state
                logits = vl_gpt.gen_head(hidden_states[:, -1, :])

                # ch: ------ #
                # probs = torch.softmax(logits / temperature, dim=-1)
                # next_token = torch.multinomial(probs, num_samples=1)
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                # ch: ------ #

                generated_tokens[:, i] = next_token.squeeze(dim=-1)
                next_token = next_token.view(-1)
                img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
                cur_inputs_embeds = img_embeds.unsqueeze(dim=1)
                inputs_embeds = torch.cat([inputs_embeds, cur_inputs_embeds], dim=1)

            dec = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
            dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
            dec = np.clip((dec + 1) / 2 * 255, 0, 255)
            visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
            visual_img[:, :, :] = dec
            for i in range(parallel_size):
                save_path = os.path.join(pre_image_dir, f'step_{step}.png')
                PIL.Image.fromarray(visual_img[i]).save(save_path)

        # ### ------generate action "mode 1" -------- #####
        # generate_ids = vl_gpt.language_model.generate(inputs_embeds=inputs_embeds, max_new_tokens=7)
        # print(generate_ids)

        # ### ------generate action "mode 2" -------- #####
        # stacked_img_embeds = torch.cat(img_embeds_list, dim=1)
        # inputs_embeds = torch.cat([inputs_embeds, stacked_img_embeds], dim=1)
        # generated_action_tokens = torch.zeros((parallel_size, action_token_num), dtype=torch.int).to(device)

        # add_tokens = torch.tensor([vl_chat_processor.image_end_id]*generated_action_tokens.shape[0]).to(device)
        # add_embeds = vl_gpt.language_model.get_input_embeddings()(add_tokens).unsqueeze(1)
        # inputs_embeds = torch.cat([inputs_embeds, add_embeds], dim=1)
        
        # for i in range(action_token_num):
        #     if i == 0:
        #         outputs = vl_gpt.language_model.model(
        #             inputs_embeds=inputs_embeds, 
        #             use_cache=True,
        #             image_indexes=torch.arange(0, inputs_embeds.shape[1]).to(device),
        #             action_indexes=torch.arange(0, 0).to(device)
        #         )
        #     else:
        #         outputs = vl_gpt.language_model.model(
        #             inputs_embeds=cur_inputs_embeds, 
        #             use_cache=True,
        #             past_key_values=outputs.past_key_values,
        #             image_indexes=torch.arange(0, 0).to(device),
        #             action_indexes=torch.arange(0, 1).to(device)
        #         )

        #     hidden_states = outputs.last_hidden_state
        #     logits = vl_gpt.language_model.lm_head(hidden_states[:, -1, :])

        #     # ch: ------ #
        #     # probs = torch.softmax(logits / temperature, dim=-1)
        #     # next_token = torch.multinomial(probs, num_samples=1)
        #     next_token = torch.argmax(logits, dim=-1, keepdim=True)
        #     # ch: ------ #

        #     generated_action_tokens[:, i] = next_token.squeeze(dim=-1)
        #     next_token = next_token.view(-1)
        #     action_emb = vl_gpt.language_model.get_input_embeddings()(next_token)
        #     cur_inputs_embeds = action_emb.unsqueeze(dim=1)
        # print(generated_action_tokens)


        ### ------generate action "mode 2" -------- #####
        generated_action_tokens = torch.zeros((parallel_size, action_token_num), dtype=torch.int).to(device)
        add_tokens = torch.tensor([vl_chat_processor.image_end_id]*generated_action_tokens.shape[0]).to(device)
        add_embeds = vl_gpt.language_model.get_input_embeddings()(add_tokens).unsqueeze(1)
        inputs_embeds = torch.cat([inputs_embeds, add_embeds], dim=1)
        
        images_len = inputs_embeds.shape[1]
        for i in range(action_token_num):
            outputs = vl_gpt.language_model.model(
                inputs_embeds=inputs_embeds, 
                image_indexes=torch.arange(0, images_len).to(device),
                action_indexes=torch.arange(images_len, images_len+i).to(device)
            )

            hidden_states = outputs.last_hidden_state
            logits = vl_gpt.language_model.lm_head(hidden_states[:, -1, :])

            # ch: ------ #
            # probs = torch.softmax(logits / temperature, dim=-1)
            # next_token = torch.multinomial(probs, num_samples=1)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            # ch: ------ #

            generated_action_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = next_token.view(-1)
            action_emb = vl_gpt.language_model.get_input_embeddings()(next_token)
            cur_inputs_embeds = action_emb.unsqueeze(dim=1)
            inputs_embeds = torch.cat([inputs_embeds, cur_inputs_embeds], dim=1)
        print(generated_action_tokens)

        normalized_actions = action_tokenizer.decode_token_ids_to_actions(generated_action_tokens.cpu().numpy())
        if normalized_actions.ndim == 1:
            dim = len(normalized_actions)
            if dim == 7 or dim == 14:
                normalized_actions[6] = 0 if normalized_actions[6] < 0.5 else 1
            if dim == 14:
                normalized_actions[13] = 0 if normalized_actions[13] < 0.5 else 1
        else:
            dim = normalized_actions.shape[1]
            if dim == 7 or dim == 14:
                normalized_actions[:, 6] = (normalized_actions[:, 6] >= 0.5).astype(int)
            if dim == 14:
                normalized_actions[:, 13] = (normalized_actions[:, 13] >= 0.5).astype(int)


        actions = np.where(
            statistic['action_mask'],
            0.5 * (normalized_actions + 1) * (statistic['action_max'] - statistic['action_min']) + statistic['action_min'],
            normalized_actions,
        )


        return actions

def model_predict_mask_once_kvcache(args, vl_gpt, vl_chat_processor, action_tokenizer, statistic, task_description, image, state, pointcloud, pre_image_dir, step):
    device = f'cuda:{args.cuda}'
    vl_gpt = vl_gpt.to(device).eval()
    parallel_size=1
    img_len = 1
    temperature = 1.0
    image_token_num_per_image = 576
    action_token_num = 7
    img_size = 384
    patch_size = 16

    state_tokens = ""
    if args.use_robot_state:
        state = np.array(state, dtype=np.float32)
        normalized_state = np.where(
            statistic['state_mask'],
            np.clip(2 * (state - statistic['state_min']) / (statistic['state_max'] - statistic['state_min'] + 1e-8) - 1, -1, 1),
            state
        )
        state_tokens += action_tokenizer(normalized_state)


    input_img_tokens_2 = vl_chat_processor.image_start_tag + vl_chat_processor.pad_tag*vl_chat_processor.num_image_tokens +vl_chat_processor.image_end_tag
    user_content = input_img_tokens_2 * img_len + task_description + state_tokens + vl_chat_processor.image_start_tag

    conversation = [
                    {"role": "<|User|>","content": user_content},
                    # {"role": "<|Assistant|>", "content": ""}
                ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )

    with torch.inference_mode():
        input_image_pixel_values = vl_chat_processor.image_processor(image, return_tensors="pt")['pixel_values'].to(torch.bfloat16).to(device)
        quant_input, emb_loss_input, info_input = vl_gpt.gen_vision_model.encode(input_image_pixel_values)
        image_tokens_input = info_input[2].detach().reshape(input_image_pixel_values.shape[0], -1)
        image_embeds_input = vl_gpt.prepare_gen_img_embeds(image_tokens_input)

        input_ids =  torch.LongTensor(vl_chat_processor.tokenizer.encode(sft_format))
        tokens = torch.zeros((parallel_size, len(input_ids)), dtype=torch.long).to(device)

        for i in range(parallel_size):
            tokens[i, :] = input_ids
        image_embeds_input = image_embeds_input.repeat(parallel_size, 1, 1)

        # torch.set_printoptions(threshold=10_000)
        # print(tokens)
        # print(image_embeds_input.shape)
        
        tokens[tokens < 0] = 0  # ignore the image embeddings
        inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
        image_gen_indices = (tokens == vl_chat_processor.image_start_id).nonzero()
        if args.image_generation:
            image_gen_indices = [
                ind for ii, ind in enumerate(image_gen_indices) 
                if (ii + 1) % 2 != 0
            ]
        for in_img_index, ind in enumerate(image_gen_indices):
            offset = ind[1] + 1
            inputs_embeds[ind[0], offset:offset+image_embeds_input.shape[1], :] = image_embeds_input[in_img_index]
        
        action_condition_len = inputs_embeds.shape[1]-1
        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)
        generated_action_tokens = torch.zeros((parallel_size, action_token_num), dtype=torch.int).to(device)
        for i in range(image_token_num_per_image+action_token_num):

            if i<image_token_num_per_image:
                outputs = vl_gpt.language_model.model(
                    inputs_embeds=inputs_embeds, 
                    image_indexes=torch.arange(0, inputs_embeds.shape[1]).to(device),
                    action_indexes=torch.arange(0, 0).to(device),
                    image_generation = args.image_generation,
                    use_cache=True, 
                    past_key_values=outputs.past_key_values if i!=0 else None
                )
                hidden_states = outputs.last_hidden_state
                logits = vl_gpt.gen_head(hidden_states[:, -1, :])

                # ch: ------ #
                # probs = torch.softmax(logits / temperature, dim=-1)
                # next_token = torch.multinomial(probs, num_samples=1)
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                # ch: ------ #

                generated_tokens[:, i] = next_token.squeeze(dim=-1)
                next_token = next_token.view(-1)
                img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
                inputs_embeds = img_embeds.unsqueeze(dim=1)

            else:
                if i == image_token_num_per_image:
                    add_tokens = torch.tensor([vl_chat_processor.image_end_id]*parallel_size).to(device)
                    add_embeds = vl_gpt.language_model.get_input_embeddings()(add_tokens).unsqueeze(1)
                    inputs_embeds = torch.cat([inputs_embeds, add_embeds], dim=1)

                    outputs = vl_gpt.language_model.model(
                        inputs_embeds=inputs_embeds, 
                        image_indexes=torch.arange(0, inputs_embeds.shape[1]).to(device),
                        action_indexes=torch.arange(0, 0).to(device),
                        action_condition_len=action_condition_len,
                        image_generation = args.image_generation,
                        use_cache=True, 
                        past_key_values=outputs.past_key_values
                    )
                else:
                    outputs = vl_gpt.language_model.model(
                        inputs_embeds=inputs_embeds, 
                        image_indexes=torch.arange(0, 0).to(device),
                        action_indexes=torch.arange(0, inputs_embeds.shape[1]).to(device),
                        image_generation = args.image_generation,
                        action_condition_len=action_condition_len,
                        use_cache=True, 
                        past_key_values=outputs.past_key_values
                    )

                hidden_states = outputs.last_hidden_state
                # logits = vl_gpt.language_model.lm_head_action(hidden_states[:, -1, :])
                logits = vl_gpt.language_model.lm_head(hidden_states[:, -1, :])

                # ch: ------ #
                # probs = torch.softmax(logits / temperature, dim=-1)
                # next_token = torch.multinomial(probs, num_samples=1)
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                # ch: ------ #

                generated_action_tokens[:, i-image_token_num_per_image] = next_token.squeeze(dim=-1)
                next_token = next_token.view(-1)
                # action_emb = vl_gpt.language_model.get_input_embeddings_action()(next_token)
                action_emb = vl_gpt.language_model.get_input_embeddings()(next_token)
                inputs_embeds = action_emb.unsqueeze(dim=1)

        print(generated_action_tokens)
        dec = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)
        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec
        for i in range(parallel_size):
            save_path = os.path.join(pre_image_dir, f'step_{step}.png')
            PIL.Image.fromarray(visual_img[i]).save(save_path)

        normalized_actions = action_tokenizer.decode_token_ids_to_actions(generated_action_tokens.cpu().numpy())
        if normalized_actions.ndim == 1:
            dim = len(normalized_actions)
            if dim == 7 or dim == 14:
                normalized_actions[6] = 0 if normalized_actions[6] < 0.5 else 1
            if dim == 14:
                normalized_actions[13] = 0 if normalized_actions[13] < 0.5 else 1
        else:
            dim = normalized_actions.shape[1]
            if dim == 7 or dim == 14:
                normalized_actions[:, 6] = (normalized_actions[:, 6] >= 0.5).astype(int)
            if dim == 14:
                normalized_actions[:, 13] = (normalized_actions[:, 13] >= 0.5).astype(int)
        actions = np.where(
            statistic['action_mask'],
            0.5 * (normalized_actions + 1) * (statistic['action_max'] - statistic['action_min']) + statistic['action_min'],
            normalized_actions,
        )
        return actions
    
def main(args):
    # Report the arguments
    Logger.log_info(f'Running {colored(__file__, "red")} with arguments:')
    Logger.log_info(f'task name: {args.task_name}')
    Logger.log_info(f'number of episodes: {args.num_episodes}')
    Logger.log_info(f'result directory: {args.result_dir}')
    Logger.log_info(f'exp name: {args.exp_name}')
    Logger.log_info(f'actions chunk: {args.action_chunk}')
    Logger.log_info(f'replay or predict: {args.replay_or_predict}')
    Logger.log_info(f'max steps: {args.max_steps}')
    Logger.log_info(f'cuda used: {args.cuda}')
    # cprint('-' * os.get_terminal_size().columns, 'cyan')

    action_mode = RLBenchActionMode.eepose_then_gripper_action_mode(absolute=True)
    obs_config = RLBenchObservationConfig.single_view_config(camera_name='front', image_size=(224, 224))
    env = RLBenchEnv(
        task_name=args.task_name,
        action_mode=action_mode,
        obs_config=obs_config,
        point_cloud_camera_names=['front'],
        cinematic_record_enabled=True,
        num_points=1024,
        use_point_crop=True,
    )
    env = VideoWrapper(env)
    
    if args.replay_or_predict == 'predict':
        args.result_dir = os.path.join(args.result_dir, 'predict_results')
    elif args.replay_or_predict == 'replay':
        args.result_dir = os.path.join(args.result_dir, 'replay_results')
    
    if args.exp_name is None:
        args.exp_name = args.task_name

    video_dir = os.path.join(
        args.result_dir, args.task_name, args.exp_name, "videos"
    )
    recreate_directory(video_dir)
    
    log_dir = os.path.join(
        os.path.join(
            args.result_dir, args.task_name, args.exp_name
        ),
        f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    recreate_directory(log_dir)
    logger = setup_logger(log_dir)

    success_num = 0
    # #----------- for model predict
    if args.replay_or_predict == 'predict':
        vl_gpt, vl_chat_processor, action_tokenizer, statistic = model_load(args)
        episode_length = args.max_steps

    for i in range(args.num_episodes):

        pre_image_dir = os.path.join(
            args.result_dir, args.task_name, args.exp_name, "pre_image", f"episode{i}"
        )
        recreate_directory(pre_image_dir)

        #----------- for key frames replay
        if args.replay_or_predict == 'replay':
            dat = np.load(os.path.join(args.replay_data_dir, args.task_name, f'episode{i}.npy'),allow_pickle = True)
            task_description = dat[0]['language_instruction']
            episode_length = len(dat)

        logger.info(f'episode: {i}, steps: {episode_length}')
        obs_dict = env.reset()
        terminated = False
        success = False
        gripper_open = None
        
        for j in range(episode_length):
            
            # #--------- for key frames replay
            if args.replay_or_predict == 'replay':
                action = dat[j]['action']
                robo_state = dat[j]['state']

                
                sum_first_3_rows = np.sum(action.reshape(8, 7)[:, :3], axis=0)
                last_row_last_4 = action[-4:]
                action = np.concatenate([sum_first_3_rows, last_row_last_4])

                # print(action[3:6],robo_state[3:6])
                # print(action[3:6],robo_state[3:6])


                action[:3] += robo_state[:3]
                gripper_open = action[-1]
                action = EEpose.pose_6DoF_to_7DoF(action[:-1])
                action = np.append(action, gripper_open)
                print(j, "  :", action)
                obs_dict, reward, terminated, truncated, info = env.step(action)
                Image.fromarray(obs_dict['image']).save(f"/gpfs/0607-cluster/chenhao/step{j}.png")
                success = success or bool(reward)

                
            # # #----------- for model predict
            if args.replay_or_predict == 'predict':
                image = obs_dict['image']
                image = [Image.fromarray(image)]
                task_description = env.text
                robot_state = obs_dict['robot_state']
                robot_state = EEpose.pose_7DoF_to_6DoF(robot_state[7:14])
                robot_state = np.concatenate([robot_state, np.array([gripper_open])]) if gripper_open != None else np.concatenate([robot_state, np.array([1])])
                cur_robot_state = robot_state if args.use_robot_state else None

                if args.load_pointcloud:
                    point_cloud = obs_dict['point_cloud']
                else:
                    point_cloud=None

                actions = model_predict_mask_once_kvcache(args, vl_gpt, vl_chat_processor, action_tokenizer, statistic, task_description, image, cur_robot_state, point_cloud, pre_image_dir, step = j)

                for action in actions:
                    action[:3] += obs_dict['robot_state'][7:10]
                    gripper_open = action[-1]
                    action = EEpose.pose_6DoF_to_7DoF(action[:-1])
                    action = np.append(action, gripper_open)
                    logger.info("%d  : %s", j, action)
                    obs_dict, reward, terminated, truncated, info = env.step(action)
                    success = success or bool(reward)
                    if terminated or truncated or success:
                        break

                if terminated or truncated or success:
                    break
                
        if success:
            success_num += 1

        image_dir = os.path.join(
            args.result_dir, args.task_name, args.exp_name, "images", f"episode{i}"
        )
        recreate_directory(image_dir)

        env.save_video(os.path.join(video_dir, f'episode{i}_video_steps.mp4'))
        env.save_images(image_dir, quiet=True)
        logger.info(f'episode{i}_{success}')
        Logger.print_seperator()
    
    logger.info(f'Finished. {args.task_name} * {args.num_episodes}. Success rate {success_num/args.num_episodes*100}%')
    with open(os.path.join(args.result_dir, args.task_name, f'{args.exp_name}_success_rate.txt'), "w", encoding="utf-8") as file:
        file.write(f'Finished. {args.task_name} * {args.num_episodes}. Success rate {success_num/args.num_episodes*100}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-name', type=str, default='close_box')
    parser.add_argument('--dataset-name', type=str, default='rlbench')
    parser.add_argument('--replay-or-predict', type=str, default='predict')
    parser.add_argument('--num-episodes', type=int, default=20)
    parser.add_argument('--result-dir', type=str, default='./result')
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--max-steps', type=int, default=10)
    parser.add_argument('--cuda', type=str, default='7')
    parser.add_argument('--use_robot_state', type=int, default=1)
    parser.add_argument('--load-pointcloud', type=int, default=0)
    parser.add_argument('--action-chunk', type=int, default=1)
    parser.add_argument('--image_generation', type=int, default=0, help='generate image')
    parser.add_argument('--replay_data_dir', type=str, default='/gpfs/0607-cluster/chenhao/data/rlbench/keyframe_fast_slow_chunk8_addlast_0806/for_rlds')
    main(parser.parse_args())