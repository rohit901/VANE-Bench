# VANE-Bench: Video Anomaly Evaluation Benchmark for Conversational LMMs

[Rohit Bharadwaj*](https://rohit901.github.io), [Hanan Gani*](https://hananshafi.github.io/), [Muzammal Naseer](https://muzammal-naseer.com/), [Fahad Khan](https://sites.google.com/view/fahadkhans/home), [Salman Khan](https://salman-h-khan.github.io/)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()

Official code for our paper "VANE-Bench: Video Anomaly Evaluation Benchmark for Conversational LMMs"

## :rocket: News
* **(June 11, 2024)**
  * Our code, data, and the [project website](https://hananshafi.github.io/vane-benchmark/) is now live!

<hr>


![method-diagram](https://hananshafi.github.io/vane-benchmark/static/images/Main%20VANE-Bench%20Flow_v7.png)
> **Abstract:** *The recent developments in Large Multi-modal Video Models (Video-LMMs) have significantly enhanced our ability to interpret and analyze video data. Despite their impressive capabilities, current Video-LMMs have not been evaluated for anomaly detection tasks, which is critical to their deployment in practical scenarios e.g., towards identifying deepfakes, manipulated video content, traffic accidents and crimes. In this paper, we introduce VANE-Bench, a benchmark designed to assess the proficiency of Video-LMMs in detecting and localizing anomalies and inconsistencies in videos. Our dataset comprises an array of videos synthetically generated using existing state-of-the-art text-to-video generation models, encompassing a variety of subtle anomalies and inconsistencies grouped into five categories: unnatural transformations, unnatural appearance, pass-through, disappearance and sudden appearance. Additionally, our benchmark features real-world samples from existing anomaly detection datasets, focusing on crime-related irregularities, atypical pedestrian behavior, and unusual events. The task is structured as a visual question-answering challenge to gauge the models' ability to accurately detect and localize the anomalies within the videos. We evaluate nine existing Video-LMMs, both open and closed sources, on this benchmarking task and find that most of the models encounter difficulties in effectively identifying the subtle anomalies. In conclusion, our research offers significant insights into the current capabilities of Video-LMMs in the realm of anomaly detection, highlighting the importance of our work in evaluating and improving these models for real-world applications. Our project website is live at [link](https://hananshafi.github.io/vane-benchmark/)*
>

## :trophy: Key Contributions:

- We present **VANE-Bench: Video ANomaly Evaluation Benchmark**, consisting of 325 video clips, and 559 challenging question-answer pairs from both real-world video surveillance, and AI-generated videos.
- We perform detailed evaluation of over **nine state-of-the-art closed-source and open-source Video-LMMs** on VANE-Bench, and show that most models exhibit poor performance, highlighting the challenging nature of our proposed benchmark.
- We conduct detailed result analysis, and also perform human evaluation on VANE-Bench to set a reasonable benchmark target.
- We open-source our code, and describe the data construction process of VANE-Bench along with making our data publicly available.

## :hammer_and_wrench: Setup and Installation

## :floppy_disk: Datasets

## :email: Contact
Should you have any questions, please create an issue in this repository or contact rohit.bharadwaj@mbzuai.ac.ae or hanan.ghani@mbzuai.ac.ae.

## :pray: Acknowledgement

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
