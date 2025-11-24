import os, sys, pathlib
import argparse
import tqdm
import shutil
import torch
from transformers import AutoModelForCausalLM
import json
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
from pathlib import Path
from termcolor import cprint, colored

import logging
import time
from datetime import datetime

import numpy as np
import pickle

import torch
from dataclasses import dataclass
from PIL import Image



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



def model_predict_mask_once_kvcache(args, vl_gpt, vl_chat_processor, action_tokenizer, statistic, task_description, image, pre_image_dir, step, state=None):
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
                        action_condition_len=action_condition_len,
                        image_generation = args.image_generation,
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
    
    

def model_predict_mask_kvcache(args, vl_gpt, vl_chat_processor, action_tokenizer, statistic, task_description, image, pre_image_dir, step, state=None):
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

        action_inputs_embeds = inputs_embeds[:,:-1]
        image_inputs_embeds = inputs_embeds


        ### ------generate action "mode 2" -------- #####
        generated_action_tokens = torch.zeros((parallel_size, action_token_num), dtype=torch.int).to(device)

        condition_len = action_inputs_embeds.shape[1]
        cache_position_condition = torch.arange(0, condition_len, device=inputs_embeds.device, dtype=torch.long)
        cache_position_action = torch.arange(condition_len+577, condition_len+578, device=inputs_embeds.device, dtype=torch.long)

        add_tokens = torch.tensor([vl_chat_processor.image_end_id]*parallel_size).to(device)
        add_embeds = vl_gpt.language_model.get_input_embeddings()(add_tokens).unsqueeze(1)
        action_inputs_embeds = torch.cat([action_inputs_embeds, add_embeds], dim=1)

        print(action_inputs_embeds.shape, condition_len)
        for i in range(action_token_num):
            if i == 0:
                outputs = vl_gpt.language_model.model(
                    inputs_embeds=action_inputs_embeds, 
                    cache_position=torch.cat([cache_position_condition, cache_position_action]),
                    image_indexes=torch.arange(0, condition_len+1).to(device),
                    action_indexes=torch.arange(0, 0).to(device),
                    use_cache=True, 
                    past_key_values=None,
                    image_generation = False,
                )
            else:
                outputs = vl_gpt.language_model.model(
                    inputs_embeds=action_inputs_embeds, 
                    cache_position=torch.arange(condition_len+577+i, condition_len+578+i, device=inputs_embeds.device, dtype=torch.long),
                    image_indexes=torch.arange(0, 0).to(device),
                    action_indexes=torch.arange(0, action_inputs_embeds.shape[1]).to(device),
                    use_cache=True, 
                    past_key_values=outputs.past_key_values,
                    image_generation = False,
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
            action_inputs_embeds = action_emb.unsqueeze(dim=1)
        print(generated_action_tokens)


        ### ------generate image-------- #####
        if args.image_generation:
            generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)
            for i in range(image_token_num_per_image):
                if i == 0:
                    outputs = vl_gpt.language_model.model(
                        inputs_embeds=image_inputs_embeds, 
                        image_indexes=torch.arange(0, image_inputs_embeds.shape[1]).to(device),
                        action_indexes=torch.arange(0, 0).to(device),
                        use_cache=True, 
                        past_key_values=None,
                        image_generation = True,
                    )
                else:
                    outputs = vl_gpt.language_model.model(
                        inputs_embeds=image_inputs_embeds, 
                        image_indexes=torch.arange(0, image_inputs_embeds.shape[1]).to(device),
                        action_indexes=torch.arange(0, 0).to(device),
                        use_cache=True, 
                        past_key_values=outputs.past_key_values,
                        image_generation = True,
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
                image_inputs_embeds = img_embeds.unsqueeze(dim=1)

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
    

def model_predict_mask(args, vl_gpt, vl_chat_processor, action_tokenizer, statistic, task_description, image, pre_image_dir, step, state=None):
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

        action_inputs_embeds = inputs_embeds[:,:-1]
        image_inputs_embeds = inputs_embeds

        ### ------generate action "mode 2" -------- #####
        generated_action_tokens = torch.zeros((parallel_size, action_token_num), dtype=torch.int).to(device)

        condition_len = action_inputs_embeds.shape[1]
        cache_position_condition = torch.arange(0, condition_len, device=inputs_embeds.device, dtype=torch.long)

        add_tokens = torch.tensor([vl_chat_processor.image_end_id]*parallel_size).to(device)
        add_embeds = vl_gpt.language_model.get_input_embeddings()(add_tokens).unsqueeze(1)
        action_inputs_embeds = torch.cat([action_inputs_embeds, add_embeds], dim=1)
        
        for i in range(action_token_num):
            cache_position_action = torch.arange(condition_len+577, action_inputs_embeds.shape[1]+577, device=inputs_embeds.device, dtype=torch.long)
            outputs = vl_gpt.language_model.model(
                inputs_embeds=action_inputs_embeds, 
                cache_position=torch.cat([cache_position_condition, cache_position_action]),
                image_indexes=torch.arange(0, condition_len).to(device),
                action_indexes=torch.arange(condition_len, action_inputs_embeds.shape[1]).to(device)
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
            action_inputs_embeds = torch.cat([action_inputs_embeds, cur_inputs_embeds], dim=1)

        print(generated_action_tokens)

        # if args.image_generation:
        #     generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)
        #     for i in range(image_token_num_per_image):
        #         outputs = vl_gpt.language_model.model(
        #             inputs_embeds=image_inputs_embeds, 
        #             image_indexes=torch.arange(0, image_inputs_embeds.shape[1]).to(device),
        #             action_indexes=torch.arange(0, 0).to(device)
        #         )
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
        #         image_inputs_embeds = torch.cat([image_inputs_embeds, cur_inputs_embeds], dim=1)

        #     dec = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
        #     dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        #     dec = np.clip((dec + 1) / 2 * 255, 0, 255)
        #     visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        #     visual_img[:, :, :] = dec
        #     for i in range(parallel_size):
        #         save_path = os.path.join(pre_image_dir, f'step_{step}.png')
        #         PIL.Image.fromarray(visual_img[i]).save(save_path)

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



def model_predict_mask_once(args, vl_gpt, vl_chat_processor, action_tokenizer, statistic, task_description, image, pre_image_dir, step, state=None):
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

        image_emb_len = inputs_embeds.shape[1]
        if args.image_generation:
            generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)
            generated_action_tokens = torch.zeros((parallel_size, action_token_num), dtype=torch.int).to(device)
            for i in range(image_token_num_per_image+action_token_num):

                if i<image_token_num_per_image:
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

                else:
                    if i == image_token_num_per_image:
                        add_tokens = torch.tensor([vl_chat_processor.image_end_id]*parallel_size).to(device)
                        add_embeds = vl_gpt.language_model.get_input_embeddings()(add_tokens).unsqueeze(1)
                        inputs_embeds = torch.cat([inputs_embeds, add_embeds], dim=1)
                    outputs = vl_gpt.language_model.model(
                        inputs_embeds=inputs_embeds,
                        image_indexes=torch.arange(0, image_emb_len+image_token_num_per_image).to(device),
                        action_indexes=torch.arange(image_emb_len+image_token_num_per_image, inputs_embeds.shape[1]).to(device)
                    )
                    hidden_states = outputs.last_hidden_state
                    logits = vl_gpt.language_model.lm_head(hidden_states[:, -1, :])
                    # ch: ------ #
                    # probs = torch.softmax(logits / temperature, dim=-1)
                    # next_token = torch.multinomial(probs, num_samples=1)
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    # ch: ------ #
                    generated_action_tokens[:, i-576] = next_token.squeeze(dim=-1)
                    next_token = next_token.view(-1)
                    action_emb = vl_gpt.language_model.get_input_embeddings()(next_token)
                    cur_inputs_embeds = action_emb.unsqueeze(dim=1)
                    inputs_embeds = torch.cat([inputs_embeds, cur_inputs_embeds], dim=1)
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
    vl_gpt, vl_chat_processor, action_tokenizer, statistic = model_load(args)
    # task_description = "Pick up the two Lego bricks from the green board and place them beside the board."
    
    with open('/gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/json/lego_new_1500_three_test.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    os.makedirs(args.result_dir, exist_ok=True)

    for sample in data:
        task_description = sample['input_prompt']
        image_paths = sample['input_image']
        p = Path(image_paths[0])
        result = f"{p.parent.name}_{p.name}"
        result = result.replace(".png", "_pred.png")

        image = [PIL.Image.open(image_path).convert("RGB").resize((384, 384), Image.BICUBIC) for image_path in image_paths]
        actions = model_predict(args, vl_gpt, vl_chat_processor, action_tokenizer, statistic, task_description, image, pre_image_dir=args.result_dir, step=result)
        print(actions)


def main_single(args):
    vl_gpt, vl_chat_processor, action_tokenizer, statistic = model_load(args)
    task_description = "put carrot on plate"
    
    os.makedirs(args.result_dir, exist_ok=True)

    image_paths = [
        "/gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/simpler/PutOnPlateInScene25Single-v1_500/success_proc_0_numid_0_epsid_0/image13.png",
    ]

    image = [PIL.Image.open(image_path).convert("RGB") for image_path in image_paths]
    actions = model_predict_mask_once_kvcache(args, vl_gpt, vl_chat_processor, action_tokenizer, statistic, task_description, image, pre_image_dir=args.result_dir, step=3)
    print(actions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default='real_world')
    parser.add_argument('--num-episodes', type=int, default=20)
    parser.add_argument('--result-dir', type=str, default='./result')
    parser.add_argument('--model-path', type=str, default='/gpfs/0607-cluster/chenhao/DoubleRL-VLA/exp/image_only/janus_pro_no_siglip_encoder_1B_2e-5/checkpoint-99-1500')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--max-steps', type=int, default=20)
    parser.add_argument('--action-dim', type=int, default=14)
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--use_robot_state', type=int, default=0)
    parser.add_argument('--load-pointcloud', type=int, default=0)
    parser.add_argument('--action-chunk', type=int, default=1)
    parser.add_argument('--image_generation', type=int, default=1, help='generate image')
    main_single(parser.parse_args())