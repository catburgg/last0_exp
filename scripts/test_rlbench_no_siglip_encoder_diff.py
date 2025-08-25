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


def unique_euler_xyz_rad(angles, range_style="2pi"):
    """
    输入: 欧拉角 (xyz 顺序)，弧度制 (任意范围, 可正可负, 可超过 2π)
    输出: 欧拉角 (xyz 顺序)，弧度制，严格唯一表示

    参数:
        precision: 保留小数位数
        range_style: "negpi" -> (-π, π], "2pi" -> [0, 2π)
    """
    # 输入是弧度
    rot = R.from_euler('xyz', angles, degrees=False)
    
    # 转回 xyz (弧度制)
    euler = rot.as_euler('xyz', degrees=False)
    
    # wrap 到 (-π, π]
    euler = (euler + np.pi) % (2 * np.pi) - np.pi
    
    # 约束: y ∈ [-π/2, π/2]
    if euler[1] > np.pi/2:
        euler[1] = np.pi - euler[1]
        euler[0] += np.pi
        euler[2] += np.pi
    elif euler[1] < -np.pi/2:
        euler[1] = -np.pi - euler[1]
        euler[0] += np.pi
        euler[2] += np.pi
    
    # 再 wrap 一次
    euler = (euler + np.pi) % (2 * np.pi) - np.pi
    
    # 如果要求 [0, 2π)，再转换
    if range_style == "2pi":
        euler = euler % (2 * np.pi)
    
    return euler


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
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        diff=True, action_dim=7
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
    num_ddim_steps = args.ddim_steps

    state_tokens = ""
    if args.use_robot_state:
        state = np.array(state, dtype=np.float32)
        normalized_state = np.where(
            statistic['state_mask'],
            np.clip(2 * (state - statistic['state_min']) / (statistic['state_max'] - statistic['state_min'] + 1e-8) - 1, -1, 1),
            state
        )
        state_tokens += action_tokenizer(normalized_state)


    input_img_tokens_1 = vl_chat_processor.image_start_tag + vl_chat_processor.image_tag*vl_chat_processor.num_image_tokens +vl_chat_processor.image_end_tag
    input_img_tokens_2 = vl_chat_processor.image_start_tag + vl_chat_processor.pad_tag*vl_chat_processor.num_image_tokens +vl_chat_processor.image_end_tag
    pre_data = []
    user_content = input_img_tokens_2 * img_len + task_description + state_tokens

    conversation = [
                    {"role": "<|User|>","content": user_content},
                    {"role": "<|Assistant|>", "content": ""}
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
        
        # torch.set_printoptions(threshold=10_000)
        # print(tokens)

        tokens[tokens < 0] = 0  # ignore the image embeddings
        inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)

        image_gen_indices = (tokens == vl_chat_processor.image_start_id).nonzero()
        for in_img_index, ind in enumerate(image_gen_indices):
            offset = ind[1] + 1
            inputs_embeds[ind[0], offset:offset+image_embeds_input.shape[1], :] = image_embeds_input[in_img_index]

        
        noise = torch.randn(inputs_embeds.shape[0], args.action_chunk, 7, device=device)
        sample_fn = vl_gpt.forward_diff
        model_kwargs = {'inputs_embeds': inputs_embeds}
        if num_ddim_steps is not None:
            if vl_gpt.ddim_diffusion is None:
                vl_gpt.create_ddim(ddim_step=num_ddim_steps)
            samples = vl_gpt.ddim_diffusion.ddim_sample_loop(
                sample_fn, 
                noise.shape, 
                noise, 
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=device,
                eta=0.0
            )
        else:
            raise ValueError("no ddim steps!")
        
        normalized_actions = samples[0].cpu().numpy()
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


        if args.image_generation:
            # --------------generate image------------ #

            add_tokens = [100001, 100016] if '7B' in args.model_path else [100001, 100003]
            add_tokens = torch.cat([generate_ids, torch.tensor([add_tokens]*generate_ids.shape[0]).to(device)], dim=-1)
            add_embeds = vl_gpt.language_model.get_input_embeddings()(add_tokens)
            inputs_embeds = torch.cat([inputs_embeds, add_embeds], dim=1)


            generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)
            for i in range(image_token_num_per_image):
                outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
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

            dec = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
            dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

            dec = np.clip((dec + 1) / 2 * 255, 0, 255)

            visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
            visual_img[:, :, :] = dec

            for i in range(parallel_size):
                save_path = os.path.join(pre_image_dir, f'step_{step}.png')
                PIL.Image.fromarray(visual_img[i]).save(save_path)

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
    cprint('-' * os.get_terminal_size().columns, 'cyan')

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
                # action[3:6] = unique_euler_xyz_rad(action[3:6])
                # robo_state[3:6] = unique_euler_xyz_rad(robo_state[3:6])
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
                robot_state[3:6] = unique_euler_xyz_rad(robot_state[3:6])
                cur_robot_state = robot_state if args.use_robot_state else None

                if args.load_pointcloud:
                    point_cloud = obs_dict['point_cloud']
                else:
                    point_cloud=None

                actions = model_predict(args, vl_gpt, vl_chat_processor, action_tokenizer, statistic, task_description, image, cur_robot_state, point_cloud, pre_image_dir, step = j)

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
    parser.add_argument('--ddim-steps', type=int, default=10)
    parser.add_argument('--cuda', type=str, default='7')
    parser.add_argument('--use_robot_state', type=int, default=1)
    parser.add_argument('--load-pointcloud', type=int, default=0)
    parser.add_argument('--action-chunk', type=int, default=1)
    parser.add_argument('--image_generation', type=int, default=0, help='generate image')
    parser.add_argument('--replay_data_dir', type=str, default='/gpfs/0607-cluster/chenhao/data/rlbench/keyframe_fast_slow_chunk8_addlast_0806/for_rlds')
    main(parser.parse_args())