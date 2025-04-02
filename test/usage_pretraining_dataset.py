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

from data.pretraining_datasetV2 import get_chexpert_multimodal_dataloader
from utils.logging_utils import LoggingManager


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True


def main():
    set_seed()
    config = OmegaConf.load("./test/usage_pretraining_dataset.yaml")
    #config = OmegaConf.load("./test/usage_chexpert_5x200_dataset.yaml")

    # Print configuration using the logging utility
    LoggingManager.print_config(config, "Configuration")

    # Create dataloader
    data_loader, dataset = get_chexpert_multimodal_dataloader(config, split="train")

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
        print(f"  Sample caption: {caption_text}")
        
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
Sentence lengths: min=2, mean=12.43, max=90 [p5=3.00, p95=26.00]
Sentences per report: min=0, mean=1.09, max=3 [p5=1.00, p95=2.00]
Processed 190927 valid reports, removed 144 invalid paths
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
  Caption lengths: tensor([29, 27, 21, 19, 18, 17, 17, 17, 16, 15, 14, 14, 14, 12,  9,  9])
  Example path: .train/patient37393/study9/view1_frontal.jpg
  Sample caption: [CLS] one view upright chest radiograph with poststernotomy chest redemonstrates right upper lobe volume loss and right upper lobe opacity 
  [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
Batch 2:
  Images shape: torch.Size([16, 3, 224, 224])
  Caption IDs shape: torch.Size([16, 97])
  Token type IDs shape: torch.Size([16, 97])
  Attention mask shape: torch.Size([16, 97])
  Caption lengths: tensor([34, 26, 25, 24, 24, 22, 20, 18, 15, 15, 14, 13, 12, 11, 11,  8])
  Example path: .train/patient23897/study1/view1_frontal.jpg
  Sample caption: [CLS] frontal and lateral views of the chest demonstrate opacities in the bilateral lower lobes left greater than right concerning for consolidation for which superimposed infection cannot be excluded 
  [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]


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

### 1. `[CLS]` - Classification Token
- Placed at the beginning of every input sequence
- Used to aggregate sequence-level information
- For classification tasks, the final hidden state of this token is used as the aggregate representation
- In your multimodal learning setup, this token likely helps the model relate image features to text features

### 2. `[SEP]` - Separator Token
- Marks the boundary between different segments of text
- In your case, it signals the end of the actual report text

### 3. `[PAD]` - Padding Token
- Used to make all sequences in a batch the same length
- Notice in your code that all captions are padded to a fixed length (97 tokens as configured in the YAML)
- The model's attention mechanism ignores these tokens through the attention mask

Here's what's happening:
1. The raw text from reports is processed and tokenized
2. Special tokens `[CLS]` and `[SEP]` are added
3. The sequence is padded to a fixed length with `[PAD]` tokens
4. The decoded output includes all of these special tokens
"""