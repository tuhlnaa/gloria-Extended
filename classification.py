import os

import torch
from configs.config import parse_args
import gloria
import pandas as pd

from gloria import builder 

CHEXPERT_5x200 = r"D:\Kai\pretrained\Gloria\chexpert_5x200.csv"
checkpoint_path = r"D:\Kai\pretrained\Gloria\chexpert_resnet50.ckpt"
config = parse_args()

df = pd.read_csv(CHEXPERT_5x200)
df = df[0:1]
full_paths = [os.path.join(config.data_dir, path.replace('CheXpert-v1.0/', '')) for path in df['Path']]

# load model
gloria_model = builder.build_gloria_model(config).to(config.device.device)

checkpoint = torch.load(checkpoint_path, map_location=config.device.device)
model_state_dict = builder.normalize_model_state_dict(checkpoint)
gloria_model.load_state_dict(model_state_dict)

class_prompts = gloria.generate_chexpert_class_prompts()
print(class_prompts)

"""
{
'Atelectasis': ['minimal residual atelectasis at the left lung zone', 'minimal subsegmental atelectasis at the left lung base', ' trace atelectasis at the mid lung zone', 'mild bandlike atelectasis at the lung bases', 'minimal bandlike atelectasis at the right lung base'], 
'Cardiomegaly': [' portable view of the chest demonstrates mild cardiomegaly ', ' cardiac silhouette size is upper limits of normal ', ' heart size remains at mildly enlarged ', ' mildly prominent cardiac silhouette ', ' ap erect chest radiograph demonstrates the heart size is the upper limits of normal '], 
'Consolidation': ['apperance of bilateral consolidation at the right lung base', 'improved patchy consolidation at the lower lung zone', 'apperance of partial consolidation at the left upper lobe', 'increased partial consolidation at the left lung base', 'increased airspace consolidation at the upper lung zone'], 
'Edema': [' pulmonary edema ', 'improvement in pulmonary interstitial edema ', 'decreased pulmonary edema ', 'moderate pulmonary edema ', 'mild pulmonary edema '], 
'Pleural Effusion': ['increased left subpulmonic pleural effusion', 'stable tiny bilateral pleural effusion', 'large tiny subpulmonic pleural effusion', 'decreased tiny subpulmonic pleural effusion', ' tiny bilateral pleural effusion']}
"""

# process input images and class prompts 
processed_txt = gloria_model.process_class_prompts(class_prompts, config.device.device)
processed_imgs = gloria_model.process_images(full_paths, config.device.device)
print(processed_imgs.shape)  # torch.Size([5, 3, 224, 224])

# zero-shot classification on 1000 images
similarities = gloria.zero_shot_classification(
    gloria_model, processed_imgs, processed_txt)
print(similarities)
"""
   Atelectasis  Cardiomegaly  Consolidation     Edema  Pleural Effusion
0      0.14921      0.213611       0.165815  0.153986          0.142162
"""

labels = df[gloria.constants.CHEXPERT_COMPETITION_TASKS].to_numpy().argmax(axis=1)
pred = similarities[gloria.constants.CHEXPERT_COMPETITION_TASKS].to_numpy().argmax(axis=1)
acc = len(labels[labels == pred]) / len(labels) 

print(df[gloria.constants.CHEXPERT_COMPETITION_TASKS].to_numpy())
# [[1. 0. 0. 0. 0.]]

# print(labels)
# # [0]
# print(similarities[gloria.constants.CHEXPERT_COMPETITION_TASKS].to_numpy())
# # [[0.1492098  0.21361108 0.16581504 0.15398553 0.14216161]]
# print(pred)
# # [1]
print(acc)
# 0.0
# 0~100 :0.25
       
"""
python classification.py --config configs\default_gloria_config.yaml

C:/Users/<user>/.cache/huggingface/hub/models--emilyalsentzer--Bio_ClinicalBERT/snapshots/d5892b39a4adaed74b92212a44081509db72f87b/pytorch_model.bin
C:/Users/<user>/.cache/huggingface/hub/models--emilyalsentzer--Bio_ClinicalBERT/snapshots/587607fe30b99405d51a27a47254de2b66763a8f/model.safetensors
C:/Users/<user>/.cache/huggingface/hub/models--emilyalsentzer--Bio_ClinicalBERT/snapshots/d5892b39a4adaed74b92212a44081509db72f87b/vocab.txt
C:/Users/<user>/.cache/huggingface/hub/models--emilyalsentzer--Bio_ClinicalBERT/snapshots/d5892b39a4adaed74b92212a44081509db72f87b/config.json
"""