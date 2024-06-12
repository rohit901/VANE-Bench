## Generating Predictions for VANE-Bench dataset from Video-LMMs
This document provides detailed instructions on generating answers (predictions) from Video-LMMs for VQA in the VANE-Bench. 


**How the predictions are generated?** For each QA pair of VANE-Bench, we feed Video-LMM with the question with 4 options alongside with the corresponding video, which picks one of the 4 options as the predicted response. Each QA pair is processed without maintaining the chat history.

Below we provide separate instructions for each model for generating predictions.

### Generating Predictions using TimeChat
Follow the steps below to generate answers for TimeChat Video-LMM

1) First install the required packages and create the environment for TimeChat by following instructions in [INSTALL.md](INSTALL.md).

2) Run the following commands to generate predictions for TimeChat using standard prompting (i.e. by asking the question only)
```shell
cd VANE-Bench/scripts/open_source_models/TimeChat
```

```shell
# --dataset_path is the path that points to the dataset in downloaded VANE-Bench folder
# --output_dir can be any path to save the predictions
CUDA_VISIBLE_DEVICES=0 python inference_vanebench.py \
--dataset_path <path-to-dataset-in-VANE-Bench-folder> \
--output_dir <folder-path-to-save-predictions>
```


### Generating Predictions using Video-LLaVA
Follow the steps below to generate answers for Video-LLaVA Video-LMM

1) First install the required packages and create the environment for Video-LLaVA by following instructions in [INSTALL.md](INSTALL.md).

2) Run the following commands to generate predictions for Video-LLaVA using standard prompting (i.e. by asking the question only)
```shell
cd VANE-Bench/scripts/open_source_models/Video-LLaVA
conda activate videollava
```

```shell
# --dataset_path is the path that points to the dataset in downloaded VANE-Bench folder
# --output_dir can be any path to save the predictions
CUDA_VISIBLE_DEVICES=0 python videollava/eval/inference_vanebench.py \
--dataset_path <path-to-dataset-in-VANE-Bench-folder> \
--output_dir <folder-path-to-save-predictions>
```

### Generating Predictions using Video-LLaMA
Follow the steps below to generate answers for Video-LLaMA Video-LMM

1) Use the environment of Video-LLaVA above and follow the instructions in [INSTALL.md](INSTALL.md).

2) Run the following commands to generate predictions for Video-LLaMA using standard prompting (i.e. by asking the question only)
```shell
cd VANE-Bench/scripts/open_source_models/Video-LLaMA
conda activate videollava
```

```shell
# --dataset_path is the path that points to the dataset in downloaded VANE-Bench folder
# --output_dir can be any path to save the predictions
CUDA_VISIBLE_DEVICES=0 python inference_vanebench.py --cfg-path ./eval_configs/video_llama_eval_only_vl.yaml \
--dataset_path <path-to-dataset-in-VANE-Bench-folder> \
--output_dir <folder-path-to-save-predictions>
```
