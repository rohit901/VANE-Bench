# VANE-Bench: Video Anomaly Evaluation Benchmark for Conversational LMMs

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Image">
</p>


## Getting started with VANE-Bench on open-source models

### Download VANE-Bench Dataset

VANE-Bench dataset can be downloaded [using this link (zipped)](https://drive.google.com/drive/folders/1DkmPlSy2naUCyw0AA2NYYUUsWFbAM0cF).
After unzipping, the VANE-Bench dataset structure looks like the following:

```
VQA_Data/
|–– Real World/
|   |–– UCFCrime
|   |   |–– Arrest002 
|   |   |–– Arrest002_qa.txt
|   |   |–– ... # remaining video-qa pairs
|   |–– UCSD-Ped1
|   |   |–– Test_004 
|   |   |–– Test_004_qa.txt
|   |   |–– ... # remaining video-qa pairs
... # remaining real-world anomaly dataset folders
|–– AI-Generated/
|   |–– SORA
|   |   |–– video_1_subset_2 
|   |   |–– video_1_subset_2_qa.txt
|   |   |–– ... # remaining video-qa pairs
|   |–– opensora
|   |   |–– 1 
|   |   |–– 1_qa.txt
|   |   |–– ... # remaining video-qa pairs
... # remaining AI-generated anomaly dataset folders
```

Here, each folder corresponds to a single video dataset and contains annotations (QA pairs) alongside video frames. 


### Evaluating Video-LMMs on VANE-Bench
To evaluate Video-LMMs on the VANE-Bench, please follow the following steps:

#### 0) Installation
Follow the instructions in [INSTALL.md](assets/INSTALL.md) to install packages and model weights required to run the sample Video-LMM codes for evaluation. 

#### 1) Generating Predictions for VANE-Bench dataset from Video-LMMs

For each QA pair with multiple options, we prompt the Video-LMMs to generate a single option (A, B C, or D) corresponding to the correct answer. Follow [PREDICTIONS.md](assets/PREDICTIONS.md) for sample codes for generating answers using TimeChat, Video-LLaVA, Video-LLaMA. 

#### 2) Comparing the Predicted Answers with Ground-Truth Answers for evaluation
Once the answer predictions are generated from step 1, we calculate the standard accuracy by giving a score of 1 if the correct option matches the ground truth option and 0 if the answer is wrong. Use the script [calc_lmm_vqa_accuracy.py](https://github.com/rohit901/VANE-Bench/blob/main/scripts/calc_lmm_vqa_accuracy.py) for computing evaluation accuracy. 
```shell
  python calc_lmm_vqa_accuracy.py --path <path-to-prediction-file>

```

<hr />

## License
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. The videos in VANE-Bench dataset are either generated using open-source Generative models or using existing publicly available anomaly datasets and are for academic research use only. 
By using VANE-Bench, you agree not to use the dataset for any harm or unfair discrminiation. Please note that the data in this dataset may be subject to other agreements. Video copyrights belong to the original dataset providers, video creators, or platforms.




## Acknowledgements

This repository has borrowed Video-LMM evaluation code from [CVRR-Evaluation-Suite](https://github.com/mbzuai-oryx/CVRR-Evaluation-Suite/), [TimeChat](https://github.com/RenShuhuai-Andy/TimeChat) and [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID). We also borrowed partial code from [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT) repository. We thank the authors for releasing their code.

