import re
import os
from typing import Optional
import numpy as np
import pandas as pd
import cv2
import tqdm
import pickle
import numpy.random as random
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


from PIL import Image
from nltk.tokenize import RegexpTokenizer
from transformers import AutoTokenizer
from data.dataset import build_transformation
from gloria.constants import *


class MultimodalPretrainingDataset(data.Dataset):
    def __init__(self, config, split="train", transform=None):
        
        # Set column name mappings
        self.data_dir = Path(config.data_dir)
        self.path_col = config.dataset.columns.path
        self.view_col = config.dataset.columns.view
        self.report_col = config.dataset.columns.report
        self.split_col = config.dataset.columns.split

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = config
        self.transform = transform
        self.max_word_num = self.cfg.dataset.text.captions_per_image

        # read CheXpert csv file
        self.csv_path = os.path.join(config.data_dir, config.master_csv)
        self.df = pd.read_csv(self.csv_path)

        # self.df[self.path_col] = self.df[self.path_col].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        self.df = self._load_chexpert_dataframe()
        self.df = self.df[self.df[self.view_col] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)


    def _load_chexpert_dataframe(self) -> pd.DataFrame:
        """Load and preprocess the CheXpert CSV file."""
        print(f"Loading CSV from: {self.csv_path}")
            
        df = pd.read_csv(self.csv_path)
        
        # Convert relative paths to absolute paths if needed
        if not os.path.isabs(df[self.path_col].iloc[0]):
            df[self.path_col] = df[self.path_col].apply(
                lambda x: os.path.join(self.data_dir, x)
            )
        
        print(f"Loaded dataframe with {len(df)} rows and columns: {df.columns.tolist()}")
        return df


    def load_text_data(self, split):

        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[self.split_col] == split][
            self.path_col
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        return filenames, path2sent

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.dataset.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.dataset.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.dataset.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        return imgs, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[self.report_col]) == str:
                captions += row[self.report_col]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[self.path_col])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[self.path_col]] = study_sent
            else:
                to_remove.append(row[self.path_col])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img


def multimodal_collate_fn(batch):
    """sort sequence"""

    imgs, cap_len, ids, tokens, attention, path = [], [], [], [], [], []

    # flattern
    for b in batch:
        img, cap, cap_l, p = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)

    # stack
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)
    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "paths": path,
    }

    return return_dict


def get_chexpert_multimodal_dataloader(
        config,
        split: str = "train",
        transform: Optional[T.Compose] = None, 
    ) -> DataLoader:
    """
    Create a DataLoader for the CheXpert medical multimodal dataset.
    
    Args:
        config: Configuration object containing dataset parameters
            Expected structure:
            - config.data_dir: Base directory containing CheXpert data
            - config.{split}_csv: Path to the specific split CSV file
            - config.model.batch_size: Batch size
            - config.dataset.num_workers: Number of workers
            - config.dataset.fraction: Optional fraction of data to use (for training)
        split: Dataset split ('train', 'valid', or 'test')
        transform: Optional custom transformation pipeline; if None, uses transforms built from config
        
    Returns:
        DataLoader for the CheXpert dataset
    """
    # Create transformation if not provided
    if transform is None:
        transform = build_transformation(config, split)
    
    # Create dataset
    dataset = MultimodalPretrainingDataset(
        config=config,
        split=split,
        transform=transform,
    )
    
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty! No images found. Please check the paths and file formats.")
    
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

    return data_loader, dataset