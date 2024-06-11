# Import the Python SDK
import google.generativeai as genai
import os
import ast
import glob
import argparse

from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from PIL import Image

def main(data_path, out_path):

    load_dotenv()

    genai.configure(api_key = os.environ['GOOGLE_API_KEY'])

    safety_settings = {
        'SEXUALLY_EXPLICIT': 'block_none',
        'HATE_SPEECH': 'block_none',
        'HARASSMENT': 'block_none',
        'DANGEROUS': 'block_none',
    }

    generation_config = genai.types.GenerationConfig(
        max_output_tokens = 2,
        temperature = 0
    )

    system_instruction = "You are a helpful and intelligent multi-modal AI assistant, capable of performing visual question answering (VQA) task. You will be given as input 10 consecutive frames from a video, and a corresponding question related to the video, you have to answer the given question after analyzing and understanding the given input video. The question itself will present you with 4 lettered options like A) B) C) D), your task is to only output single letter corresponding to the correct answer (i.e. string literal 'A', 'B', 'C', or 'D'), and you should not output anything else."

    model = genai.GenerativeModel(model_name = 'gemini-1.5-pro-latest', system_instruction = system_instruction)
    Path(out_path).mkdir(parents=True, exist_ok=True)

    dirs = os.listdir(data_path)
    dirs = [d for d in dirs if os.path.isdir(os.path.join(data_path, d))]
    dirs = sorted(dirs, key=lambda x: (int(x.split('_')[1]), int(x.split('_')[3]))) # sort directories which are named in the format: "video_1_subset_2"

    f_names = dirs
    dirs = [os.path.join(data_path, d) for d in dirs]

    for idx, dir_path in tqdm(enumerate(dirs), total=len(dirs)):
        print("="*50)
        print(f"Processing {f_names[idx]}")
        txt_path = os.path.join(data_path, f_names[idx] + "_qa.txt")
        img_paths = glob.glob(dir_path + "/*.jpg")
        img_paths = sorted(img_paths, key = lambda x: int(x.split("/")[-1].split("_")[-1].split(".")[0])) # sort frames which are named in the format: "frame_137.jpg", important to ensure that the frames are in the correct (ascending) order

        with open(txt_path, "r") as f:
            qa_data = f.read().strip()
            qa_data = ast.literal_eval(qa_data)

        preds_path = os.path.join(out_path, f_names[idx] + "_preds.txt")
        img_list = [Image.open(img_path) for img_path in img_paths]

        for qa in qa_data:
            question = qa["Q"]
            GT_answer = qa["A"]

            if GT_answer[-1] == ')':
                GT_answer = GT_answer[:-1]

            response = model.generate_content([*img_list, question], stream = True, safety_settings = safety_settings, generation_config = generation_config)
            response.resolve()

            with open(preds_path, "a") as f:
                f.write(f"{GT_answer}\t{response.text}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True, help="Path to the directory containing the raw dataset, i.e., put absolute path to either 'SORA', 'ModelScope', 'UCFCrime', etc which will contain multiple subdirectories each containing the raw frames of a video.")
    parser.add_argument("--out_path", type=str, required=True, help="Path to the directory where the predictions will be saved.")
    args = parser.parse_args()

    main(args.data_path, args.out_path)