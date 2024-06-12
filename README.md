# VANE-Bench: Video Anomaly Evaluation Benchmark for Conversational LMMs

[Rohit Bharadwaj*](https://rohit901.github.io), [Hanan Gani*](https://hananshafi.github.io/), [Muzammal Naseer](https://muzammal-naseer.com/), [Fahad Khan](https://sites.google.com/view/fahadkhans/home), [Salman Khan](https://salman-h-khan.github.io/)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]() [![Dataset](https://img.shields.io/badge/Dataset-Download-orange?logo=database)](https://huggingface.co/datasets/rohit901/VANE-Bench) [![Website](https://img.shields.io/badge/Website-Visit-green?logo=web)](https://hananshafi.github.io/vane-benchmark/)


Official code for our paper "VANE-Bench: Video Anomaly Evaluation Benchmark for Conversational LMMs"

*Authors marked with "\*" contributed equally to this work.

## :rocket: News
* **(June 11, 2024)**
  * Our code, [data](https://huggingface.co/datasets/rohit901/VANE-Bench), and the [project website](https://hananshafi.github.io/vane-benchmark/) is now live!

<hr>


![method-diagram](https://github.com/rohit901/VANE-Bench/blob/main/assets/Main_VANE-Bench%20Flow_v7.png?raw=true)
> **Abstract:** *The recent developments in Large Multi-modal Video Models (Video-LMMs) have significantly enhanced our ability to interpret and analyze video data. Despite their impressive capabilities, current Video-LMMs have not been evaluated for anomaly detection tasks, which is critical to their deployment in practical scenarios e.g., towards identifying deepfakes, manipulated video content, traffic accidents and crimes. In this paper, we introduce VANE-Bench, a benchmark designed to assess the proficiency of Video-LMMs in detecting and localizing anomalies and inconsistencies in videos. Our dataset comprises an array of videos synthetically generated using existing state-of-the-art text-to-video generation models, encompassing a variety of subtle anomalies and inconsistencies grouped into five categories: unnatural transformations, unnatural appearance, pass-through, disappearance and sudden appearance. Additionally, our benchmark features real-world samples from existing anomaly detection datasets, focusing on crime-related irregularities, atypical pedestrian behavior, and unusual events. The task is structured as a visual question-answering challenge to gauge the models' ability to accurately detect and localize the anomalies within the videos. We evaluate nine existing Video-LMMs, both open and closed sources, on this benchmarking task and find that most of the models encounter difficulties in effectively identifying the subtle anomalies. In conclusion, our research offers significant insights into the current capabilities of Video-LMMs in the realm of anomaly detection, highlighting the importance of our work in evaluating and improving these models for real-world applications. Our project website is live at [link](https://hananshafi.github.io/vane-benchmark/)*
>

## :trophy: Key Contributions:

- We present **VANE-Bench: Video ANomaly Evaluation Benchmark**, consisting of 325 video clips, and 559 challenging question-answer pairs from both real-world video surveillance, and AI-generated videos.
- We perform detailed evaluation of over **nine state-of-the-art closed-source and open-source Video-LMMs** on VANE-Bench, and show that most models exhibit poor performance, highlighting the challenging nature of our proposed benchmark.
- We conduct detailed result analysis, and also perform human evaluation on VANE-Bench to set a reasonable benchmark target.
- We open-source our code, and describe the data construction process of VANE-Bench along with making our data publicly available.

## :hammer_and_wrench: Setup and Usage
To replicate our experiments, and to use our code:
1. First clone the repository:
```bash
git clone git@github.com:rohit901/VANE-Bench.git
```
or
```bash
git clone https://github.com/rohit901/VANE-Bench.git
```
2. Change directory:
```bash
cd VANE-Bench
```

### Closed-Source LMMs Setup and Usage
We used `python=3.11.8` in our experiments involving closed-source LMMs like GPT-4o and Gemini-1.5 Pro. 
1. Setup a new conda environment with the specified python version:
```bash
conda create --name vane_bench python=3.11
```
2. Activate the environment
```bash
conda activate vane_bench
```
3. Install Libraries:
```bash
pip install openai opencv-python python-dotenv tqdm google-generativeai pillow
```
4. Create a new `.env` file in `scripts/closed_source_models/`, and populate it with your OpenAI and Gemini API keys:
```bash
OPENAI_API_KEY=<your_key_here>
GOOGLE_API_KEY=<your_key_here>
```

#### Caption Generation Module (CGM)
CGM requires path to the directory containing frames annotated with bounding boxes. The `path` argument in the script is the absolute path to annotated dataset directory like 'SORA', 'ModelScope', 'UCFCrime', etc, and each of these dataset directories will contain multiple subdirectories (one per video clip in the dataset) containing the annotated frames. You can use our CGM with your own annotated data.
To run the code:
```bash
python scripts/closed_source_models/CGM.py --path="<path_to_annotated_dataset>"
```
The above script will then generate caption for each video clip in the same directory as `path`.

#### Question Answer Generation Module (QAGM)
Once the captions are obtained from CGM, we can use QAGM to generate the final QA pairs. QAGM also requires `path` to the annotated data, and assumes that the captions are also in the same path.
To run the code:
```bash
python scripts/closed_source_models/QAGM.py --path="<path_to_annotated_dataset_and_captions>"
```

#### Evaluating GPT-4o on VQA task
1. Download and unzip the VANE-Bench dataset by following [Dataset](#floppy_disk-dataset).
2. Evaluate GPT-4o one dataset at a time. For example, to evaluate it on "SORA" dataset, run the following:
```bash
python scripts/closed_source_models/evaluate_vqa_GPT.py --data_path="/path/to/VQA_Data/AI-Generated/SORA" --out_path="/path/to/GPT-4o/SORA"
```

#### Evaluating Gemini-1.5 Pro on VQA task
1. Download and unzip the VANE-Bench dataset by following [Dataset](#floppy_disk-dataset).
2. Evaluate Gemini-1.5 Pro one dataset at a time. For example, to evaluate it on "SORA" dataset, run the following:
```bash
python scripts/closed_source_models/eval_vqa_gemini.py --data_path="/path/to/VQA_Data/AI-Generated/SORA" --out_path="/path/to/Gemini-Pro/SORA"
```

#### Calculating LMMs accuracy on VQA task
Following the previous instruction, once the prediction files for a LMM is generated, we can evaluate the LMM's accuracy by running:
```bash
python scripts/calc_lmm_vqa_accuracy.py --path="/path/to/GPT-4o/SORA"
```
The above command evaluates the accuracy of GPT-4o on the SORA dataset. To evaluate different models on different datasets, just modify the `path` variable accordingly.

### Open-Source LMMs Setup and Usage
Follow the instructions [here](https://github.com/rohit901/VANE-Bench/tree/main/scripts/open_source_models) for setting up and evaluating open-source Video-LMMs.

## :floppy_disk: Dataset
Our VANE-Bench dataset can be downloaded from the following [Drive Link](https://drive.google.com/drive/folders/1DkmPlSy2naUCyw0AA2NYYUUsWFbAM0cF?usp=sharing)

## :email: Contact
Should you have any questions, please create an issue in this repository or contact rohit.bharadwaj@mbzuai.ac.ae or hanan.ghani@mbzuai.ac.ae.

## :pray: Acknowledgement
We thank [OpenAI](https://github.com/openai/openai-python) and [Google](https://github.com/google-gemini/generative-ai-python) for their Python SDKs. 

## :black_nib: Citation
If you found our work helpful, please consider starring the repository ⭐⭐⭐ and citing our work as follows:
```bibtex
@article{vane2024bharadwaj,
  author    = {Bharadwaj, Rohit and Gani, Hanan and Naseer, Muzammal and Khan, Fahad and Khan, Salman},
  title     = {VANE-Bench: Video Anomaly Evaluation Benchmark for Conversational LMMs},
  journal   = {Arxiv},
  year      = {2024},
}
```
