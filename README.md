# Noise-Guided Predicate Representation Extraction and Diffusion-Enhanced Discretization for Scene Graph Generation (NoDIS)

The implementation code for the paper "Noise-Guided Predicate Representation Extraction and Diffusion-Enhanced Discretization for Scene Graph Generation"

🎉 This paper has been accepted by ICML2025 [![PDF](https://img.shields.io/badge/Paper-PDF-orange)](./Noise_Guided_Predicate_Representation_Extraction_and_Diffusion_Enhanced_Discretization_for_Scene_Graph_Generation.pdf)

<p align="center">
  <img src="./figs/overview.pdf" width="300"/>
</p>

<p align="center">
  <img src="./figs/model_detail.pdf" width="300"/>
</p>


## ✅ TODO
- [x] upload code (Inference using DDPM)
- [x] Update the inference method and introduce the DDIM denoising method

## Installation
Check [INSTALL.md](./INSTALL.md) for installation instructions.

## Dataset
Check [DATASET.md](./DATASET.md) for instructions of dataset preprocessing.

## Train and Test
We provide [scripts](./scripts/train.sh) for training and testing the models

## Device
All our experiments are conducted on one NVIDIA GeForce RTX 3090, if you wanna run it on your own device, make sure to follow distributed training instructions in [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

## Quantitative Analysis
For the quantitative evaluation results presented in the paper, we provide the [computation code](./tools/quality_assessment.py).

## The Trained Model Weights
### After the paper is published, we will upload the code to GitHub, and it will be continuously updated. The pre-trained models are being gradually uploaded to the cloud storage.


## Tips

We use the `rel_nms` [operation](./maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py) provided by [RU-Net](https://github.com/siml3/RU-Net/blob/main/maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py) and [HL-Net](https://github.com/siml3/HL-Net/blob/main/maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py) in PredCls and SGCls to filter the predicted relation predicates, which encourages diverse prediction results. 

## Help

Be free to contact me (`#####`) if you have any questions!

## Acknowledgement

The code is implemented based on [PENet](https://github.com/VL-Group/PENET) and [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

## Citation

