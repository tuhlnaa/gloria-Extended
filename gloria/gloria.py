
import os
import torch
import copy
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import segmentation_models_pytorch as smp

from itertools import product
from typing import Dict, Literal, Union, List

from . import builder
from . import utils
from . import constants
from .models.vision_model import PretrainedImageClassifier


np.random.seed(6)
random.seed(6)


_MODELS = {
    "gloria_resnet50": "./pretrained/chexpert_resnet50.ckpt",
    "gloria_resnet18": "./pretrained/chexpert_resnet18.ckpt",
}


_SEGMENTATION_MODELS = {
    "gloria_resnet50": "./pretrained/chexpert_resnet50.ckpt",
}


_FEATURE_DIM = {"gloria_resnet50": 2048, "gloria_resnet18": 2048}


def available_models() -> List[str]:
    """Returns the names of available GLoRIA models"""
    return list(_MODELS.keys())


def available_segmentation_models() -> List[str]:
    """Returns the names of available GLoRIA models"""
    return list(_SEGMENTATION_MODELS.keys())


def load_gloria(
    name: str = "gloria_resnet50",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Load a GLoRIA model

    Parameters
    ----------
    name : str
        A model name listed by `gloria.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model

    Returns
    -------
    gloria_model : torch.nn.Module
        The GLoRIA model
    """

    # warnings
    if name in _MODELS:
        ckpt_path = _MODELS[name]
    elif os.path.isfile(name):
        ckpt_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    if not os.path.exists(ckpt_path):
        raise RuntimeError(
            f"Model {name} not found.\n"
            + "Make sure to download the pretrained weights from \n"
            + "    https://stanfordmedicine.box.com/s/j5h7q99f3pfi7enc0dom73m4nsm6yzvh \n"
            + " and copy it to the ./pretrained folder."
        )

    ckpt = torch.load(ckpt_path)
    cfg = ckpt["hyper_parameters"]
    ckpt_dict = ckpt["state_dict"]

    fixed_ckpt_dict = {}
    for k, v in ckpt_dict.items():
        new_key = k.split("gloria.")[-1]
        fixed_ckpt_dict[new_key] = v
    ckpt_dict = fixed_ckpt_dict

    gloria_model = builder.build_gloria_model(cfg).to(device)
    # Remove the problematic key
    if "text_encoder.model.embeddings.position_ids" in ckpt_dict:
        del ckpt_dict["text_encoder.model.embeddings.position_ids"]
    # if "text_encoder.model.embeddings.position_ids" in ckpt_dict:
    #     pos_ids = ckpt_dict["text_encoder.model.embeddings.position_ids"]
    #     print(f"Type: {type(pos_ids)}")
    #     print(f"Shape: {pos_ids.shape}")
    #     print(f"Data type: {pos_ids.dtype}")
    #     print(f"First few values: {pos_ids[:5, :5] if len(pos_ids.shape) > 1 else pos_ids[:5]}")
    #     print(pos_ids)
    gloria_model.load_state_dict(ckpt_dict)

    return gloria_model


def load_img_classification_model(
    name: str = "gloria_resnet50",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    num_cls: int = 1,
    freeze_encoder: bool = True,
):
    """Load a GLoRIA pretrained classification model

    Parameters
    ----------
    name : str
        A model name listed by `gloria.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model
    num_cls: int
        Number of output classes
    freeze_encoder: bool
        Freeze the pretrained image encoder

    Returns
    -------
    img_model : torch.nn.Module
        The GLoRIA pretrained image classification model
    """

    # load pretrained image encoder
    gloria_model = load_gloria(name, device)
    image_encoder = copy.deepcopy(gloria_model.img_encoder)
    del gloria_model

    # create image classifier
    feature_dim = _FEATURE_DIM[name]
    img_model = PretrainedImageClassifier(
        image_encoder, num_cls, feature_dim, freeze_encoder
    )

    return img_model


def load_img_segmentation_model(
        name: str = "gloria_resnet50",
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
    """Load a GLoRIA pretrained classification model

    Parameters
    ----------
    name : str
        A model name listed by `gloria.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model

    Returns
    -------
    img_model : torch.nn.Module
        The GLoRIA pretrained image classification model
    """

    # warnings
    if name in _SEGMENTATION_MODELS:
        ckpt_path = _SEGMENTATION_MODELS[name]
        base_model = name.split("_")[-1]
    elif os.path.isfile(name):
        ckpt_path = name
        base_model = "resnet50"  # TODO
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_segmentation_models()}"
        )

    # load base model
    seg_model = smp.Unet(base_model, encoder_weights=None, activation=None)

    # update weight
    ckpt = torch.load(ckpt_path)
    ckpt_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("gloria.img_encoder.model"):
            k = ".".join(k.split(".")[3:])
            ckpt_dict[k] = v
        ckpt_dict["fc.bias"] = None
        ckpt_dict["fc.weight"] = None
    seg_model.encoder.load_state_dict(ckpt_dict)

    return seg_model.to(device)


def zero_shot_classification(
        model: nn.Module, 
        images: torch.Tensor, 
        class_text_mapping: Dict[str, Dict[str, torch.Tensor]]
    ) -> pd.DataFrame:
    """
    Perform zero-shot classification using a GLoRIA pretrained model.
    
    Args:
        model: GLoRIA model loaded via gloria.load_models()
        images: Processed images using model.process_img
        class_text_mapping: Dictionary mapping class names to processed text embeddings.
                           Each class can have multiple associated text prompts.
    
    Returns:
        DataFrame with similarity scores between each image and class
    """
    # Calculate similarities for each class
    class_similarities = []
    class_names = []
    
    for class_name, class_text in class_text_mapping.items():
        # Compute similarities between images and text prompts for this class
        similarities = compute_similarities(model, images, class_text)
        
        # Take max similarity across all prompts for this class
        max_similarities = similarities.max(axis=1)
        
        class_similarities.append(max_similarities)
        class_names.append(class_name)
    
    # Stack all class similarities into a single array
    similarity_matrix = np.stack(class_similarities, axis=1)
    
    # Normalize similarities across classes (only for batches with multiple images)
    if similarity_matrix.shape[0] > 1:
        similarity_matrix = utils.normalize(similarity_matrix)
    
    # Convert to DataFrame for easier handling
    return pd.DataFrame(similarity_matrix, columns=class_names)


def compute_similarities(
        gloria_model,
        images: torch.Tensor,
        texts: Dict[str, torch.Tensor],
        similarity_type: Literal["global", "local", "both"] = "both"
    ) -> np.ndarray:
    """
    Compute similarities between processed images and texts.
    
    Args:
        gloria_model: GLoRIA model loaded via gloria.load_models()
        images: Processed images using gloria_model.process_img
        texts: Processed text using gloria_model.process_text
        similarity_type: Type of similarity to compute ("global", "local", or "both")
        
    Returns:
        Array of similarities between each image and text
        
    Raises:
        ValueError: If similarity_type is invalid or inputs are not processed
    """
    # Validate inputs
    if similarity_type not in ["global", "local", "both"]:
        raise ValueError(
            f"similarity_type must be one of ['global', 'local', 'both'], got {similarity_type}"
        )
    
    if isinstance(texts, (str, list)):
        raise ValueError("Text input not processed - please use gloria_model.process_text")
    
    if isinstance(images, (str, list)):
        raise ValueError("Image input not processed - please use gloria_model.process_img")
    
    
    # Extract Embedded Features
    with torch.no_grad():
        img_emb_local, img_emb_global = gloria_model.encode_images(images)
        text_emb_local, text_emb_global, _ = gloria_model.encode_text(
            texts["caption_ids"], 
            texts["attention_mask"], 
            texts["token_type_ids"]
        )

    # Compute similarities
    global_similarities = gloria_model.compute_global_similarities(img_emb_global, text_emb_global)
    local_similarities = gloria_model.compute_local_similarities(img_emb_local, text_emb_local, texts["cap_lens"])

    # Return based on requested similarity type
    if similarity_type == "global":
        return global_similarities.detach().cpu().numpy()
    elif similarity_type == "local":
        return local_similarities.detach().cpu().numpy()
    else:
        # Combine similarities
        combined_similarities = (local_similarities + global_similarities) / 2
        return combined_similarities.detach().cpu().numpy()


def generate_chexpert_class_prompts(num_prompts: int = 5) -> Dict[str, List[str]]:
    """Generate text prompts for each CheXpert classification task.
    
    This function creates combinations of severity, subtype, and location phrases
    for each CheXpert class, then randomly samples a specified number of prompts.
    
    Args:
        num_prompts: Number of prompts to generate per class.
        
    Returns:
        Dictionary mapping each class to a list of text prompts.
    """
    np.random.seed(6)
    random.seed(6)

    class_prompts = {}
    
    for class_name, attribute_dict in constants.CHEXPERT_CLASS_PROMPTS.items():
        # Extract attributes for the current class
        severities = attribute_dict.get("severity")
        subtypes = attribute_dict.get("subtype")
        locations = attribute_dict.get("location")
        
        # Generate all possible combinations of severity, subtype, and location
        all_prompts = []
        for severity, subtype, location in product(severities, subtypes, locations):
            # Skip empty prompts and clean up extra spaces
            combined = f"{severity} {subtype} {location}"#.strip()  # 🛠️
            if combined != "  ":  # Avoid empty strings
                all_prompts.append(combined)

        # Sample prompts, handling case where fewer prompts exist than requested
        sample_size = min(num_prompts, len(all_prompts))
        class_prompts[class_name] = random.sample(all_prompts, sample_size)
    
    return class_prompts
