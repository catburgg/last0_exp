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
    action_tokenizer = ActionTokenizer(tokenizer, need_to_sub=3)

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

def model_predict_mask_once_kvcache(args, vl_gpt, vl_chat_processor, action_tokenizer, statistic, task_description, image, state, pointcloud):
    device = f'cuda:{args.cuda}'
    vl_gpt = vl_gpt.to(device).eval()
    parallel_size=1
    img_len = 1
    action_token_num = 7
    num_latent_tokens = args.latent_size

    state_tokens = ""
    if args.use_robot_state:
        state = np.array(state, dtype=np.float32)
        normalized_state = np.where(
            statistic['state_mask'],
            np.clip(2 * (state - statistic['state_min']) / (statistic['state_max'] - statistic['state_min'] + 1e-8) - 1, -1, 1),
            state
        )
        state_tokens += action_tokenizer(normalized_state)

    pre_data = []
    input_img_tokens_1 = vl_chat_processor.image_start_tag + vl_chat_processor.image_tag*vl_chat_processor.num_image_tokens +vl_chat_processor.image_end_tag
    user_content = input_img_tokens_1 * img_len + task_description + state_tokens 

    conversation = [
                    {"role": "<|User|>","content": user_content},
                ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )

    with torch.inference_mode():
        input_image_pixel_values = vl_chat_processor.image_processor(image, return_tensors="pt")['pixel_values'].to(torch.bfloat16)

        input_ids =  torch.LongTensor(vl_chat_processor.tokenizer.encode(sft_format))
        latent_start_value = torch.tensor(100846).expand(*input_ids.shape[:-1], 1)
        latent_pad_value = torch.tensor(100847).expand(*input_ids.shape[:-1], args.latent_size)
        latent_end_value = torch.tensor(100848).expand(*input_ids.shape[:-1], 1) # not used
        input_ids = torch.cat([input_ids, latent_start_value, latent_pad_value], dim=-1)

        tokens = torch.zeros((parallel_size, len(input_ids)), dtype=torch.long)

        for i in range(parallel_size):
            tokens[i, :] = input_ids
            pre_data.append(VLChatProcessorOutput(sft_format=sft_format, pixel_values=input_image_pixel_values, input_ids=tokens[i], num_image_tokens=[vl_chat_processor.num_image_tokens] * img_len))
        prepare_inputs = vl_chat_processor.batchify(pre_data)

        inputs_embeds = vl_gpt.prepare_inputs_embeds(
            input_ids=tokens.to(device),
            pixel_values=prepare_inputs['pixel_values'].to(torch.bfloat16).to(device),
            images_emb_mask=prepare_inputs['images_emb_mask'].to(device),
            images_seq_mask=prepare_inputs['images_seq_mask'].to(device)
        )
        
        input_ids = input_ids.unsqueeze(0)
        latent_indices = (input_ids == 100847).nonzero()
        latent_lists = [[idx[1].item() for idx in latent_indices if idx[0] == i] for i in range(input_ids.shape[0])]
        kv_cache_cot = None
        next_compute_range = (0, latent_indices[:, 1].min().item())
        generated_action_tokens = torch.zeros((parallel_size, action_token_num), dtype=torch.int).to(device)
        
        # print("input_ids:", input_ids)
        # print("shape of input ids and latent_indices:", input_ids.shape, latent_indices.shape)
        # print("latent_indices:", latent_indices)
        # input("Press Enter to continue...")

        # inference for latent cot embeddings
        for latent_i in range(num_latent_tokens):
            curr_inputs_embeds = inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :]
            outputs = vl_gpt.language_model.model(
                inputs_embeds=curr_inputs_embeds,
                latent_indexes=torch.arange(0, curr_inputs_embeds.shape[1]).to(device),
                action_indexes=torch.arange(0, 0).to(device),
                use_latent=args.use_latent,
                use_cache=True,
                past_key_values=kv_cache_cot if latent_i!=0 else None # for kv cache
            )
            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if latent_i + 1 >= num_latent_tokens
                    else next_compute_range[1] + 1
                ),
            )
            hidden_states = outputs[0][:, -1:, :]
            assert hidden_states.shape[1] == 1
            kv_cache_cot = outputs.past_key_values
            filling_indices = [
                (instance_idx, mask_list[latent_i])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > latent_i
            ]
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair  
                tensor_list[batch_idx][token_idx] = hidden_states[batch_idx][0]
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # inference for action tokens
        for action_i in range(action_token_num):
            if action_i==0: # the first token requires latent expert and action expert to collaborate
                add_tokens = torch.tensor([100848]*parallel_size).to(device)
                add_embeds = vl_gpt.language_model.get_input_embeddings()(add_tokens).unsqueeze(1)
                last_cot_embeds = inputs_embeds[:, -1:, :]
                inputs_embeds = torch.cat([last_cot_embeds, add_embeds], dim=1)
                # print("Using latent expert for the first action token")
                # print(inputs_embeds.shape)
                # input("Press Enter to continue...")

                outputs = vl_gpt.language_model.model(
                    inputs_embeds=inputs_embeds, 
                    latent_indexes=torch.arange(0,inputs_embeds.shape[1]-1).to(device),
                    action_indexes=torch.arange(inputs_embeds.shape[1]-1,inputs_embeds.shape[1]).to(device),
                    use_latent=args.use_latent,
                    use_cache=True, 
                    past_key_values=outputs.past_key_values,
                )
            else: # the tokens left use the action expert only
                outputs = vl_gpt.language_model.model(
                    inputs_embeds=inputs_embeds, 
                    latent_indexes=torch.arange(0, 0).to(device),
                    action_indexes=torch.arange(0, inputs_embeds.shape[1]).to(device),
                    use_latent=args.use_latent,
                    use_cache=True, 
                    past_key_values=outputs.past_key_values
                )
            
            hidden_states = outputs.last_hidden_state
            logits = vl_gpt.language_model.lm_head(hidden_states[:, -1, :])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_action_tokens[:, action_i] = next_token.squeeze(dim=-1)
            next_token = next_token.view(-1)
            action_emb = vl_gpt.language_model.get_input_embeddings()(next_token)
            inputs_embeds = action_emb.unsqueeze(dim=1)

        print("generated action tokens:", generated_action_tokens)
        # input()

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
    
    args.result_dir = os.path.join(args.result_dir, 'predict_results')
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
    vl_gpt, vl_chat_processor, action_tokenizer, statistic = model_load(args)
    episode_length = args.max_steps

    for i in range(args.num_episodes):

        pre_image_dir = os.path.join(
            args.result_dir, args.task_name, args.exp_name, "pre_image", f"episode{i}"
        )
        recreate_directory(pre_image_dir)

        logger.info(f'episode: {i}, steps: {episode_length}')
        obs_dict = env.reset()
        terminated = False
        success = False
        gripper_open = None
        
        for j in range(episode_length):
            
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

            actions = model_predict_mask_once_kvcache(args, vl_gpt, vl_chat_processor, action_tokenizer, statistic, task_description, image, cur_robot_state, point_cloud)

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
    parser.add_argument('--num-episodes', type=int, default=20)
    parser.add_argument('--result-dir', type=str, default='./result')
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--max-steps', type=int, default=10)
    parser.add_argument('--cuda', type=str, default='7')
    parser.add_argument('--use_robot_state', type=int, default=1)
    parser.add_argument('--load-pointcloud', type=int, default=0)
    parser.add_argument('--action-chunk', type=int, default=1)
    parser.add_argument('--use_latent', type=int, default=1)
    parser.add_argument('--latent_size', type=int, default=4)
    parser.add_argument('--compress_strategy',type=str, required=True,default='average')
    main(parser.parse_args())