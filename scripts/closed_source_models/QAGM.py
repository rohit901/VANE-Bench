import glob
import argparse
import os

from openai import OpenAI
from dotenv import load_dotenv
from utils import create_qa_prompt
from tqdm import tqdm

def main(path):

    load_dotenv()
      
    client = OpenAI()

    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    dirs = sorted(dirs, key=lambda x: (int(x.split('_')[1]), int(x.split('_')[3]))) # sort directories which are named in the format: "video_1_subset_2"

    f_names = dirs
    dirs = [os.path.join(path, d) for d in dirs]

    for idx, data_path in tqdm(enumerate(dirs), total=len(dirs)):
        print("="*50)
        print(f"Processing {f_names[idx]}")
        caption_file = f_names[idx] + ".txt"
        out_file = f_names[idx] + "_qa.txt"
        img_paths = glob.glob(data_path + "/*.jpg")
        img_paths = sorted(img_paths, key = lambda x: int(x.split("/")[-1].split("_")[-1].split(".")[0])) # sort frames which are named in the format: "frame_137.jpg", important to ensure that the frames are in the correct (ascending) order

        with open(os.path.join(path, caption_file), "r") as f:
            caption = f.read().strip()

        out_path = os.path.join(path, out_file)

        response = client.chat.completions.create(
          model="gpt-4o",
          messages=[
            {
                "role": "system",
                "content": "You are a helpful and intelligent AI assistant which can curate high-quality and challenging question and their corresponding answers, which are used to test the video understanding capabilities of an multi-modal LLM model capable of taking videos as their inputs."
            },
            {
              "role": "user",
              "content": create_qa_prompt(img_paths, caption)
            }
          ],
          seed = 8,
          temperature = 0,
        )

        print(response.choices[0].message.content)
        with open(out_path, "w") as f:
            f.write(response.choices[0].message.content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, required=True, help="Path to the directory containing the annotated dataset, i.e., put absolute path to either 'SORA', 'ModelScope', 'UCFCrime', etc which will contain multiple subdirectories each containing the annotated frames (with bounding box) of a video.")
    args = parser.parse_args()

    main(args.path)