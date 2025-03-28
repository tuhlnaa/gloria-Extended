import os
import gloria
import pandas as pd 

device = "cuda:1"
CHEXPERT_5x200 = r"pretrained\chexpert_5x200.csv"
img_dir = r"D:\Kai\DATA_Set_2\X-ray\CheXpert-v1.0"

df = pd.read_csv(CHEXPERT_5x200)
df = df[0:10]

full_paths = [os.path.join(img_dir, path.replace('CheXpert-v1.0/', '')) for path in df['Path']]

# load model
gloria_model = gloria.load_gloria(device=device)

# generate class prompt
# cls_promts = {
#    'Atelectasis': ['minimal residual atelectasis ', 'mild atelectasis' ...]
#    'Cardiomegaly': ['cardiomegaly unchanged', 'cardiac silhouette enlarged' ...] 
# ...
# } 

class_prompts = gloria.generate_chexpert_class_prompts()

# process input images and class prompts 
processed_txt = gloria_model.process_class_prompts(class_prompts, device)
processed_imgs = gloria_model.process_images(full_paths, device)

# zero-shot classification on 1000 images
similarities = gloria.zero_shot_classification(
    gloria_model, processed_imgs, processed_txt)

print(similarities)
labels = df[gloria.constants.CHEXPERT_COMPETITION_TASKS].to_numpy().argmax(axis=1)
pred = similarities[gloria.constants.CHEXPERT_COMPETITION_TASKS].to_numpy().argmax(axis=1)
acc = len(labels[labels == pred]) / len(labels) #0.17
print(acc)
#      Atelectasis  Cardiomegaly  Consolidation     Edema  Pleural Effusion
# 0       1.371477     -0.416303      -1.023546 -1.460464          0.145969
# 1       1.550474      0.277534       1.743613  0.187523          1.166638
# ..           ...           ...            ...       ...               ...


"""

C:/Users/<user>/.cache/huggingface/hub/models--emilyalsentzer--Bio_ClinicalBERT/snapshots/d5892b39a4adaed74b92212a44081509db72f87b/pytorch_model.bin
C:/Users/<user>/.cache/huggingface/hub/models--emilyalsentzer--Bio_ClinicalBERT/snapshots/587607fe30b99405d51a27a47254de2b66763a8f/model.safetensors
C:/Users/<user>/.cache/huggingface/hub/models--emilyalsentzer--Bio_ClinicalBERT/snapshots/d5892b39a4adaed74b92212a44081509db72f87b/vocab.txt
C:/Users/<user>/.cache/huggingface/hub/models--emilyalsentzer--Bio_ClinicalBERT/snapshots/d5892b39a4adaed74b92212a44081509db72f87b/config.json


   Atelectasis  Cardiomegaly  Consolidation     Edema  Pleural Effusion
0    -0.511970     -0.762625       1.031379  0.944730         -1.259174
1    -2.216534     -1.691719      -1.606168 -0.573690         -1.500512
2     0.456774     -0.266830       1.334458 -0.093671          0.856507
3     1.064348     -0.514511      -0.386101 -1.837835          0.570704
4     0.225337      0.549659      -0.291379  0.158554         -0.651506
5     0.582589      0.542763       0.277927  0.888471          0.409863
6    -1.276765     -0.718759      -0.522587  0.117429         -0.611307
7     1.142969     -0.069415      -0.950891 -1.060934          0.488634
8     0.181445      0.830991      -0.554048 -0.361641         -0.260467
9     0.351803      2.100447       1.667400  1.818593          1.957265


   Atelectasis  Cardiomegaly  Consolidation     Edema  Pleural Effusion
0    -0.511969     -0.762624       1.031377  0.944727         -1.718593
1    -2.216533     -1.691717      -1.606169 -0.573689         -1.144250
2     0.456773     -0.266828       1.334460 -0.093671          1.363296
3     1.064348     -0.514508      -0.386101 -1.837836         -0.031464
4     0.225338      0.549663      -0.291379  0.158553         -0.009847
5     0.582590      0.542765       0.277929  0.888471          0.523101
6    -1.276765     -0.718758      -0.522587  0.117429          0.070017
7     1.142971     -0.069413      -0.950893 -1.060933         -0.155062
8     0.181447      0.830992      -0.554046 -0.361640         -0.666549
9     0.351805      2.100448       1.667398  1.818594          1.769349
"""