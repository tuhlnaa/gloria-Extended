import os
import torch
import gloria
import pandas as pd 

device = "cuda:1"
CHEXPERT_5x200 = r"pretrained\chexpert_5x200.csv"
img_dir = r"D:\Kai\DATA_Set_2\X-ray\CheXpert-v1.0"
df = pd.read_csv(CHEXPERT_5x200)

full_paths = [os.path.join(img_dir, path.replace('CheXpert-v1.0/', '')) for path in df['Path']]

# load model
#device = "cuda" if torch.cuda.is_available() else "cpu"
gloria_model = gloria.load_gloria(device=device)

# generate class prompt
# cls_promts = {
#    'Atelectasis': ['minimal residual atelectasis ', 'mild atelectasis' ...]
#    'Cardiomegaly': ['cardiomegaly unchanged', 'cardiac silhouette enlarged' ...] 
# ...
# } 
cls_prompts = gloria.generate_chexpert_class_prompts()

# process input images and class prompts 
processed_txt = gloria_model.process_class_prompts(cls_prompts, device)
processed_imgs = gloria_model.process_img(full_paths, device)

# zero-shot classification on 1000 images
similarities = gloria.zero_shot_classification(
    gloria_model, processed_imgs, processed_txt)

print(similarities)
#      Atelectasis  Cardiomegaly  Consolidation     Edema  Pleural Effusion
# 0       1.371477     -0.416303      -1.023546 -1.460464          0.145969
# 1       1.550474      0.277534       1.743613  0.187523          1.166638
# ..           ...           ...            ...       ...               ...


"""

C:/Users/<user>/.cache/huggingface/hub/models--emilyalsentzer--Bio_ClinicalBERT/snapshots/d5892b39a4adaed74b92212a44081509db72f87b/pytorch_model.bin
C:/Users/<user>/.cache/huggingface/hub/models--emilyalsentzer--Bio_ClinicalBERT/snapshots/587607fe30b99405d51a27a47254de2b66763a8f/model.safetensors
C:/Users/<user>/.cache/huggingface/hub/models--emilyalsentzer--Bio_ClinicalBERT/snapshots/d5892b39a4adaed74b92212a44081509db72f87b/vocab.txt
C:/Users/<user>/.cache/huggingface/hub/models--emilyalsentzer--Bio_ClinicalBERT/snapshots/d5892b39a4adaed74b92212a44081509db72f87b/config.json

"""