import torch
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from tqdm import tqdm
import json
import argparse
import os

from torchvision.io import read_video
import ast
from torchvision.io import read_image
import torchvision.transforms as transforms



def parse_args():
    parser = argparse.ArgumentParser(description="vane-bench-demo")
    parser.add_argument('--dataset_path', help='Directory containing folders for a specific benchmark category.',
                        required=True)
    parser.add_argument('--output_path', help='path where you want to save all output responses.',
                        required=True)
    args = parser.parse_args(args=[])
    return args


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


my_message = "You are a helpful and intelligent multi-modal AI assistant, capable of performing visual question answering (VQA) task. You will be given video and a corresponding question related to the video as input, you have to answer the given question after analyzing and understanding the given input video. The question itself will present you with 4 lettered options like A) B) C) D), your task is to only output single letter corresponding to the correct answer (i.e. string literal 'A', 'B', 'C', or 'D'), and you should not output anything else. "

def main():
    disable_torch_init()
    model_path = 'LanguageBind/Video-LLaVA-7B'
    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = False, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    conv_mode = "llava_v1"
    print('Initializing Chat')
    args = parse_args()
    video_path = args.dataset_path
    os.makedirs(args.output_path, exist_ok=True)

    for sub_folder in os.listdir(os.path.join(video_path,"video_frames")):
        vid_frames = os.path.join(video_path,"video_frames", sub_folder)  #glob.glob(os.path.join(video_path, sub_folder,"*.avi"))
        video_tensor = frames_to_video_tensor(vid_frames)
        print(video_tensor.shape)
        json_path = os.path.join(video_path, "video_qa",sub_folder + "_qa.txt")
        with open(json_path, "r") as f:
            query = ast.literal_eval(f.read())
        all_answers = []
        # Comment the below line incase we want to see the original model response without rectification
        #inp = my_message + inp #+ message_suffix
        if type(video_tensor) is list:
            tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
        else:
            tensor = video_tensor.to(model.device, dtype=torch.float16)
        for qa_pair in query:
            conv = conv_templates[conv_mode].copy()
            roles = conv.roles
            inp = my_message + qa_pair['Q']
            print(f"{roles[1]}: {inp}")
            #inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
            inp = ' '.join([DEFAULT_IMAGE_TOKEN] * 10) + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=tensor,
                    do_sample=True,
                    temperature=0.1,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            print(f"User question is given as: {prompt}")
       
            with open(os.path.join(args.output_path, sub_folder + "_preds.txt"), "a") as f:
                f.write(qa_pair['A'] + "\t" + outputs[0] + "\n")
        

if __name__ == '__main__':
    main()