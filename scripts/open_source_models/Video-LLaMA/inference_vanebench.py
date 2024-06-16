"""
Adapted from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/demo.py
"""
import argparse
import os
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
import json
from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle,conv_llava_llama_2
import decord
import cv2
decord.bridge.set_bridge('torch')
import copy
#%%
# imports modules for registration
from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *
import ast

import re

def extract_correct_option(text):
    # Define a regular expression pattern to match the correct option (A), (B), (C), or (D)
    pattern = r'([A-D])(?:\)|\.)\s*'

    # Search for the pattern in the input text
    match = re.search(pattern, text)

    if match:
        # Extract the correct option character (group 1 of the match)
        correct_option = match.group(1)
        return correct_option
    else:
        # If no match is found, return None or handle accordingly
        return text



#%%
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_type", type=str, default='vicuna', help="The type of LLM")
    parser.add_argument('--all_dimension_folder_path', help='Directory containing folders for each benchmark category.',
                        required=False)
    parser.add_argument('--dataset_path', help='Directory containing folders for a specific benchmark category.',
                        required=True)
    parser.add_argument('--output_path', help='path where you want to save all output responses.',
                        required=True)

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================
# same prompt used for V42,,, thankfully recovered



my_message = "You are a helpful and intelligent multi-modal AI assistant, capable of performing visual question answering (VQA) task. You will be given video and a corresponding question related to the video as input, you have to answer the given question after analyzing and understanding the given input video. The question itself will present you with 4 lettered options like A) B) C) D), your task is to only output single letter corresponding to the correct answer (i.e. string literal 'A', 'B', 'C', or 'D'), and you should not output anything else. "

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()
vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')


video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
data_path = args.dataset_path
os.makedirs(args.output_path, exist_ok=True)



for video_file in os.listdir(os.path.join(data_path,"videos")):
    video_path = os.path.join(data_path,"videos", video_file)  #glob.glob(os.path.join(video_path, sub_folder,"*.avi"))
    sub_folder = video_file.split(".")[0]
    json_path = os.path.join(data_path, "video_qa",sub_folder + "_qa.txt")
    with open(json_path, "r") as f:
        query = ast.literal_eval(f.read())
    # video = vis_processor.transform(video)
    # ts.show(video.transpose(0, 1))
    
    
    img_list = []
    if args.model_type == 'vicuna':
        chat_state = default_conversation.copy()
    else:
        chat_state = conv_llava_llama_2.copy()
            
    # Upload the single video
    chat_state.system = "You are able to understand the visual content that the user provides. Ensure that the output is a single letter corresponding to the correct answer (i.e. A, B, C, or D), and you should not output anything else."

    img_list = []
    llm_message = chat.upload_video_without_audio(video_path, chat_state, img_list)   
    model_response = []
    for qa_pair in query:
        # iterate over each question
        chat_state_seperate = copy.deepcopy(chat_state)
        user_question = my_message + qa_pair['Q'] 
        # user_question = single_dict["Q"]
        num_beams = 1
        temperature = 1.0
        chat.ask(user_question, chat_state_seperate)
        try:
            llm_message = chat.answer(conv=chat_state_seperate,
                        img_list=img_list,
                        num_beams=num_beams,
                        temperature=temperature,
                        max_new_tokens=512,
                        max_length=2000)[0]
            # print(chat_state.get_prompt())
            # print(chat_state)
            outputs = llm_message.strip()
            model_response.append({"Q": user_question, "A": outputs})
            # print(llm_message)
        except Exception as e:
            print(f"Error processing video file '{video_path}': {e}")
        
        print(video_file, extract_correct_option(llm_message))
        with open(os.path.join(args.output_path, sub_folder + "_preds.txt"), "a") as f:
            f.write(qa_pair['A'] + "\t" + extract_correct_option(llm_message) + "\n")



# %%
