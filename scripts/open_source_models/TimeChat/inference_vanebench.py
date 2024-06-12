import argparse
import os
import random
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchshow as ts
from timechat.common.config import Config
from timechat.common.dist_utils import get_rank
from timechat.common.registry import registry
from timechat.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle, conv_llava_llama_2
import decord
import cv2
import time
import subprocess
from decord import VideoReader
from timechat.processors.video_processor import ToTHWC, ToUint8, load_video
decord.bridge.set_bridge('torch')

# imports modules for registration
from timechat.datasets.builders import *
from timechat.models import *
from timechat.processors import *
from timechat.runners import *
from timechat.tasks import *

import random as rnd
from transformers import StoppingCriteria, StoppingCriteriaList
from PIL import Image
import gradio as gr
import os

from torchvision.io import read_video
import ast
from torchvision.io import read_image
import torchvision.transforms as transforms
import re

def extract_correct_option(text):
    # Define a regular expression pattern to match the correct option (A), (B), (C), or (D)
    pattern = r'[A-D]\)'

    # Search for the pattern in the input text
    match = re.search(pattern, text)

    if match:
        # Extract the correct option character (remove the closing parenthesis)
        correct_option = match.group(0)[0]  # Get the first character of the matched string
        return correct_option
    else:
        # If no match is found, return None or handle accordingly
        return text

def frames_to_video_tensor(folder_path, target_size=(224, 224)):
    frame_tensors = []
    
    # Define transformation for resizing frames
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert tensor to PIL Image
        transforms.Resize(target_size),
        transforms.ToTensor()  # Convert PIL Image back to tensor
    ])
    
    # List all files in the folder
    file_names = sorted(os.listdir(folder_path))

    for file_name in file_names:
        # Check if the file is an image (e.g., JPEG, PNG, etc.)
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            # Construct the full path to the image file
            file_path = os.path.join(folder_path, file_name)
            
            # Read the image file and apply transformation
            frame_tensor = read_image(file_path)
            frame_tensor = transform(frame_tensor)
            
            # Append the transformed frame tensor to the list
            frame_tensors.append(frame_tensor)
    
    # Stack the frame tensors along a new dimension to create a video tensor
    video_tensor = torch.stack(frame_tensors)
    
    return video_tensor

#my_message = "You are a helpful and intelligent multi-modal AI assistant, capable of performing visual question answering (VQA) task. You will be given as input 10 consecutive frames from a video, and a corresponding question related to the video, you have to answer the given question after analyzing and understanding the given input video. The question itself will present you with 4 lettered options like A) B) C) D), your task is to only output single letter corresponding to the correct answer (i.e. A, B, C, or D), and you should not output anything else. Now provide an answer to the following question while adhering to the provided guidelines. Q: "

my_message = "You are a helpful and intelligent multi-modal AI assistant, capable of performing visual question answering (VQA) task. You will be given video and a corresponding question related to the video as input, you have to answer the given question after analyzing and understanding the given input video. The question itself will present you with 4 lettered options like A) B) C) D), your task is to only output single letter corresponding to the correct answer (i.e. string literal 'A', 'B', 'C', or 'D'), and you should not output anything else. "

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/timechat.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--text-query", default="What is he doing?", help="question the video")
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
    args = parser.parse_args(args=[])
    return args


print('Initializing Chat')
args = parse_args()
cfg = Config(args)

DIR="ckpt/timechat"
MODEL_DIR=f"{DIR}/timechat_7b.pth"


model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_config.ckpt = MODEL_DIR
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()

vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')
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
    for qa_pair in query:
        inp = my_message + qa_pair['Q']
        img_list = []
        chat_state = conv_llava_llama_2.copy()  # Every time, previous history will be erased.
        #chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully."
        chat_state.system = "You are able to understand the visual content that the user provides. Ensure that the output is a single letter corresponding to the correct answer (i.e. A, B, C, or D), and you should not output anything else."
        msg = chat.upload_video_without_audio(
        video_path=video_path, 
        conv=chat_state,
        img_list=img_list, 
        n_frms=96,
    )

        text_input = inp
        chat.ask(text_input, chat_state)

        num_beams = args.num_beams
        temperature = args.temperature
        llm_message = chat.answer(conv=chat_state,
                            img_list=img_list,
                            num_beams=num_beams,
                            temperature=temperature,
                            max_new_tokens=300,
                            max_length=2000)[0]

        print(video_file, extract_correct_option(llm_message))
        with open(os.path.join(args.output_path, sub_folder + "_preds.txt"), "a") as f:
            f.write(qa_pair['A'] + "\t" + extract_correct_option(llm_message) + "\n")