# GLoRIA: A Multimodal Global-Local Representation Learning Framework for Label-efficient Medical Image Recognition

[![PyTorch](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python)](https://pytorch.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![GitHub repo size](https://img.shields.io/github/repo-size/tuhlnaa/gloria-Extended?label=Repo%20size)](https://github.com/tuhlnaa/gloria-Extended)

<br>

## Abstract

[![arXiv](https://img.shields.io/badge/IEEE-ICCV48922.2021.00391-00629B?logo=ieee)](https://doi.org/10.1109/ICCV48922.2021.00391)

We propose an attentionbased framework for learning global and local representations by contrasting image sub-regions and words in the paired report. In addition, we propose methods to leverage the learned representations for various downstream medical image recognition tasks with limited labels. Our results demonstrate high-performance and label-efficiency for image-text retrieval, classification (finetuning and zeros-shot settings), and segmentation on different datasets.

> **GLoRIA: A Multimodal Global-Local Representation Learning Framework for Label-efficient Medical Image Recognition** \
> Shih-Cheng Huang (Mars), Liyue Shen, Matthew P. Lungren, Serena Yeung <br> 
> Stanford University <br>
> Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021 <br>

<br>

## Approach

<p align="center">
  <img src="https://github.com/tuhlnaa/gloria-Extended/blob/main/assets/GLoRIA%20framework.png" width="60%" alt="GLoRIA framework">
</p>
<p align="center">
  <img src="https://github.com/tuhlnaa/gloria-Extended/blob/main/assets/GLoRIA%20overview.png" width="100%" alt="GLoRIA overview">
</p>

<br>

## 📊 Performance Comparison

Pretrained weight: [GLoRIA](https://stanfordmedicine.box.com/s/j5h7q99f3pfi7enc0dom73m4nsm6yzvh), Loss: 9.8907


<br>

## 🚀 Quick Start

Start by [installing PyTorch 2.3.0](https://pytorch.org/get-started/locally/) with the right CUDA version, then clone this repository and install the dependencies.  

```bash
# Clone the repository
git clone https://github.com/tuhlnaa/gloria-Extended.git
cd gloria-Extended

# Create and activate conda environment
conda create -n gloria python=3.11
conda activate gloria

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# [Windows - PowerShell]
Get-Content requirements.txt | ForEach-Object {
    if ($_ -match "\S" -and -not $_.StartsWith("#")) {
        Write-Host "Installing $_..." -ForegroundColor Cyan
        pip install $_
    }
}

# [Linux]
while read line; do
  # Skip empty lines and comments
  if [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]]; then
    echo "Installing $line..."
    pip install $line
  fi
done < requirements.txt
```

<br>

## 💾 Datasets
To use this code, you will need to download the [CheXpert-v1.0-small](https://www.kaggle.com/datasets/ashery/chexpert), [CheXpert Plus](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1) dataset (11.0 GB) and [SIIM-ACR-Pneumothorax](https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks) dataset (4.6 GB).

```bash
python utils/download_data.py
```

After downloading the `dataset`, place the data in the dataset directory as follows:
```plain-text
└── dataset/
    ├── SIIM-ACR-Pneumothorax/
    |   ├── png_images/
    |   ├── png_masks/
    |   ├── stage_1_train_images.csv
    |   └── stage_1_test_images.csv
    └── CheXpert-v1.0-small/
        ├── train/
        ├── valid/
        ├── train.csv
        ├── valid.csv
        └── df_chexpert_plus_240401.csv
```

<br>

## 🔧 Training
```bash
# Classification
python train.py --config configs\default_config.yaml  # ImageNet Initial weight
python train.py --config configs\default_classification_optimization.yaml   # Fine-tuning hyperparameters
python train.py --config configs\default_gloria_classification_config.yaml  # GLoRIA Transfer Learning

# GLoRIA
python train.py --config configs\default_gloria_config.yaml

# Segmentation
python train.py --config configs\default_segmentation.yaml
python train.py --config configs\default_segmentation_imagenet.yaml
```

<br>

## 📊 Evaluation
```bash
python classification.py --config configs\default_gloria_config.yaml
```

<br>

## 🤝 Contributing
Contributions are welcome! If you'd like to add another solution or improve existing implementations:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingSolution`)
3. Commit your changes (`git commit -m 'Add some AmazingSolution'`)
4. Push to the branch (`git push origin feature/AmazingSolution`)
5. Open a Pull Request

<br>

## 📝 Citation

This repository is based on the following paper:

```bibtex
@inproceedings{huang2021gloria,
  title={GLoRIA: A Multimodal Global-Local Representation Learning Framework for Label-Efficient Medical Image Recognition},
  author={Huang, Shih-Cheng and Shen, Liyue and Lungren, Matthew P and Yeung, Serena},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3942--3951},
  year={2021}
}
```

**Acknowledgements**
This codebase is adapted from [ControlGAN](https://github.com/mrlibw/ControlGAN)

<br>

## 🎓 Original Work

This is an enhanced implementation of the original [GLoRIA paper](https://github.com/marshuang80/gloria). While we've modernized the codebase, all credit for the original method goes to the paper authors.

<br>

## 📮 Contact
For questions and feedback:

1. Create an issue in this repository
2. [Google docs](https://docs.google.com/forms/d/e/1FAIpQLSc7obxpa5UXQyDMLE7nssiXzg8Z5qa_kmLBZzqMuslfu8U8vQ/viewform?usp=header)
