import sys
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.dataset import build_transformation
from gloria.datasets.pretraining_datasetV2 import MultimodalPretrainingDataset, multimodal_collate_fn
from utils.logging_utils import LoggingManager


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True


def main():
    set_seed()
    config = OmegaConf.load("./test/usage_pretraining_dataset.yaml")

    # Print configuration using the logging utility
    LoggingManager.print_config(config, "Configuration")

    split = "train"
    transform = build_transformation(config, "train")
    
    # Create dataset
    dataset = MultimodalPretrainingDataset(
        config=config,
        split=split,
        transform=transform,
    )

    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty! No images found. Please check the paths and file formats.")
    print(f"Dataset created successfully with {len(dataset)} samples")

    # Create dataloader with parameters from config
    data_loader = DataLoader(
        dataset,
        batch_size=config.model.batch_size,
        shuffle=(split == "train"),
        num_workers=config.dataset.num_workers,
        pin_memory=getattr(config.model, "pin_memory", False),
        drop_last=getattr(config.model, "drop_last", False) if split == "train" else True,
        collate_fn=multimodal_collate_fn
    )
    print(f"DataLoader created successfully with {len(data_loader)} batches")

    # Test loading a few batches
    print("\nTesting batch loading:")
    for batch_idx, batch in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Images shape: {batch['imgs'].shape}")
        print(f"  Caption IDs shape: {batch['caption_ids'].shape}")
        print(f"  Token type IDs shape: {batch['token_type_ids'].shape}")
        print(f"  Attention mask shape: {batch['attention_mask'].shape}")
        print(f"  Caption lengths: {batch['cap_lens']}")
        print(f"  Example path: {batch['paths'][0]}")
        
        # Print sample caption
        caption_tokens = batch['caption_ids'][0]
        caption_text = dataset.tokenizer.decode(caption_tokens)
        print(f"  Sample caption: {caption_text[:100]}...")
        
        # Only show first 2 batches
        if batch_idx >= 1:
            break


if __name__ == "__main__":
    main()

"""
╭───────────────────────────────────┬───────────────────────────────────────────────╮
│                         Parameter │ Value                                         │
├───────────────────────────────────┼───────────────────────────────────────────────┤
│                          data_dir │ '.\CheXpert-Plus'                             │
│                        master_csv │ 'df_chexpert_plus_240401.csv'                 │
│                  model.batch_size │ 16                                            │
│              model.text.bert_type │ 'emilyalsentzer/Bio_ClinicalBERT'             │
│              dataset.dataset_name │ 'chexpert'                                    │
│                  dataset.fraction │ 1.0                                           │
│               dataset.num_workers │ 0                                             │
│              dataset.image.imsize │ 256                                           │
│             dataset.text.word_num │ 97                                            │
│   dataset.text.captions_per_image │ 5                                             │
│          dataset.text.full_report │ False                                         │
│ dataset.text.use_findings_section │ False                                         │
│              dataset.columns.path │ 'path_to_image'                               │
│              dataset.columns.view │ 'frontal_lateral'                             │
│            dataset.columns.report │ 'report'                                      │
│             dataset.columns.split │ 'split'                                       │
│    dataset.force_rebuild_captions │ False                                         │
│         dataset.debug_sample_size │ 0                                             │
│  transforms.random_crop.crop_size │ 224                                           │
╰───────────────────────────────────┴───────────────────────────────────────────────╯

Loading CSV from: ./CheXpert-Plus/df_chexpert_plus_240401.csv
Loaded dataframe with 223462 rows and columns: 
['path_to_image', 'path_to_dcm', 'frontal_lateral', 'ap_pa', 
 'deid_patient_id', 'patient_report_date_order', 'report', 
 'section_narrative', 'section_clinical_history', 'section_history', 
 'section_comparison', 'section_technique', 'section_procedure_comments', 
 'section_findings', 'section_impression', 'section_end_of_impression', 
 'section_summary', 'section_accession_number', 'age', 'sex', 'race', 
 'ethnicity', 'interpreter_needed', 'insurance_type', 'recent_bmi', 'deceased', 'split']

Caption file ./CheXpert-Plus/captions_df_chexpert_plus_240401.pickle does not exist or rebuild forced. Creating captions...
Processing reports: 100%|███████████████████████████████████████████████████████████████████| 191071/191071 [00:19<00:00, 9981.76it/s]
Sentence lengths: min=2, mean=15.86, max=112 [p5=5.00, p95=29.00]
Sentences per report: min=1, mean=1.03, max=3 [p5=1.00, p95=1.00]
Processed 191071 valid reports, removed 0 invalid paths
Saved captions to: ./CheXpert-Plus/captions_df_chexpert_plus_240401.pickle
Loaded 190869 file paths for split 'train'

Dataset created successfully with 190869 samples
DataLoader created successfully with 11930 batches

Testing batch loading:
Batch 1:
  Images shape: torch.Size([16, 3, 224, 224])
  Caption IDs shape: torch.Size([16, 97])
  Token type IDs shape: torch.Size([16, 97])
  Attention mask shape: torch.Size([16, 97])
  Caption lengths: tensor([29, 26, 24, 22, 22, 21, 21, 21, 20, 20, 19, 19, 18, 17, 14,  9])
  Example path: ./CheXpert-Plus/train/patient15104/study16/view1_frontal.jpg
  Sample caption: [CLS] narrative radiographic examination of the chest 2 8 2010 clinical history 68 years of age male...

Batch 2:
  Images shape: torch.Size([16, 3, 224, 224])
  Caption IDs shape: torch.Size([16, 97])
  Token type IDs shape: torch.Size([16, 97])
  Attention mask shape: torch.Size([16, 97])
  Caption lengths: tensor([32, 31, 26, 25, 23, 22, 21, 20, 17, 16, 15, 15, 15, 13,  9,  8])
  Example path: ./CheXpert-Plus/train/patient38792/study6/view1_frontal.jpg
  Sample caption: [CLS] narrative radiographic examination of the chest post needle biopsy 12 4 19 hours 11 31 hours c...
...

# Caption IDs:
These are numerical representations of text tokens after being processed by BERT's tokenizer. 
Each word or subword in the radiology report text is converted to a numerical ID that BERT can process.

# Token type IDs:
These are used to distinguish between different segments in a sequence. 
For example, in question-answering tasks, you might have one segment for the question and another for the context.

# Attention mask:
This is a binary mask that tells the model which tokens to "pay attention to" and which to ignore. 
It has the same shape as the token IDs (16x97) and contains 1s for actual tokens and 0s for padding tokens. 
Since sentences have different lengths, shorter sequences are padded to match the longest sequence, and the attention mask ensures that the model doesn't consider the padding when processing the text.

# Caption lengths:
This refers to the actual number of tokens in each caption before padding.

# Sample caption:
This is the actual text from the radiology report after being tokenized and then decoded back to text for display purposes. 
"""