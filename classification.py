import os
import gloria
import pandas as pd 

device = "cuda:1"
CHEXPERT_5x200 = r"pretrained\chexpert_5x200.csv"
img_dir = r"D:\Kai\DATA_Set_2\X-ray\CheXpert-v1.0"

df = pd.read_csv(CHEXPERT_5x200)
df = df[0:5]

full_paths = [os.path.join(img_dir, path.replace('CheXpert-v1.0/', '')) for path in df['Path']]

# load model
gloria_model = gloria.load_gloria(device=device)

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
processed_txt = gloria_model.process_class_prompts(class_prompts, device)
processed_imgs = gloria_model.process_images(full_paths, device)
print(processed_imgs.shape)  # torch.Size([5, 3, 224, 224])

# zero-shot classification on 1000 images
similarities = gloria.zero_shot_classification(
    gloria_model, processed_imgs, processed_txt)
print(similarities)
#      Atelectasis  Cardiomegaly  Consolidation     Edema  Pleural Effusion
# 0       1.371477     -0.416303      -1.023546 -1.460464          0.145969
# 1       1.550474      0.277534       1.743613  0.187523          1.166638
# ..           ...           ...            ...       ...               ...

labels = df[gloria.constants.CHEXPERT_COMPETITION_TASKS].to_numpy().argmax(axis=1)
pred = similarities[gloria.constants.CHEXPERT_COMPETITION_TASKS].to_numpy().argmax(axis=1)
acc = len(labels[labels == pred]) / len(labels) 
print(acc) # 0.17

       
"""
C:/Users/<user>/.cache/huggingface/hub/models--emilyalsentzer--Bio_ClinicalBERT/snapshots/d5892b39a4adaed74b92212a44081509db72f87b/pytorch_model.bin
C:/Users/<user>/.cache/huggingface/hub/models--emilyalsentzer--Bio_ClinicalBERT/snapshots/587607fe30b99405d51a27a47254de2b66763a8f/model.safetensors
C:/Users/<user>/.cache/huggingface/hub/models--emilyalsentzer--Bio_ClinicalBERT/snapshots/d5892b39a4adaed74b92212a44081509db72f87b/vocab.txt
C:/Users/<user>/.cache/huggingface/hub/models--emilyalsentzer--Bio_ClinicalBERT/snapshots/d5892b39a4adaed74b92212a44081509db72f87b/config.json
"""