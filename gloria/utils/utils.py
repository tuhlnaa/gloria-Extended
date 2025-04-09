"""
Visualization utilities for GLoRIA: A Multimodal Global-Local Representation Learning Framework
for Label-efficient Medical Image Recognition.

Adapted from: https://github.com/mrlibw/ControlGAN
"""
import torch
import torch.nn.functional as F
import numpy as np
import skimage.transform

from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Optional, Union, Any


def normalize_similarities(similarities: np.ndarray, method: str = "norm") -> np.ndarray:
    """
    Normalize similarity scores using different methods.

    Args:
        similarities: Array of similarity scores to normalize
        method: Normalization method, either "norm" or "standardize"

    Returns:
        Normalized similarity scores
    """
    if method == "norm":
        return (similarities - similarities.mean(axis=0)) / (similarities.std(axis=0) + 1e-8)
    elif method == "standardize":
        min_vals = similarities.min(axis=0)
        max_vals = similarities.max(axis=0)
        denominator = max_vals - min_vals
        # Avoid division by zero
        denominator = np.where(denominator > 0, denominator, 1)
        return (similarities - min_vals) / denominator
    else:
        raise ValueError(f"Unsupported normalization method: {method}")


def generate_color_dict(num_colors: int = 100) -> Dict[int, List[int]]:
    """Generate a dictionary of colors for visualization."""
    color_dict = {}
    
    # Base colors that repeat with some variation
    base_colors = [
        [128, 64, 128],   # purple-blue
        [244, 35, 232],   # pink
        [70, 70, 70],     # dark gray
        [102, 102, 156],  # blue-gray
        [190, 153, 153],  # pink-gray
        [153, 153, 153],  # gray
        [250, 170, 30],   # orange
        [220, 220, 0],    # yellow
        [107, 142, 35],   # olive green
        [152, 251, 152],  # light green
        [70, 130, 180],   # steel blue
        [220, 20, 60],    # crimson
        [255, 0, 0],      # red
        [0, 0, 142],      # navy
        [119, 11, 32],    # burgundy
        [0, 60, 100],     # dark blue
        [0, 80, 100],     # teal blue
        [0, 0, 230],      # blue
        [0, 0, 70],       # dark navy
        [0, 0, 0]         # black
    ]
    
    for i in range(num_colors):
        color_dict[i] = base_colors[i % len(base_colors)]
    
    return color_dict


# Visualization constants
FONT_MAX_SIZE = 50
DEFAULT_FONT_PATH = "./assets/FreeMono.ttf"
DEFAULT_FONT_SIZE = 45

# Generate color dictionary
COLOR_DICT = generate_color_dict()


def draw_caption(
        canvas: np.ndarray, 
        vis_size: int, 
        sentences: List[List[str]], 
        font_path: str = DEFAULT_FONT_PATH,
        font_size: int = DEFAULT_FONT_SIZE,
        offset_x: int = 2, 
        offset_y: int = 2
    ) -> Tuple[Image.Image, List[List[str]], List[List[int]]]:
    """
    Draw text captions on the canvas.
    
    Args:
        canvas: The image canvas to draw on
        vis_size: Visualization size for each attention map
        sentences: List of tokenized sentences
        font_path: Path to the font file
        font_size: Font size to use
        offset_x: Horizontal offset between words
        offset_y: Vertical offset for padding
    
    Returns:
        Tuple containing:
        - The image with drawn captions
        - The processed sentences
        - List of word indices for each sentence
    """
    img_txt = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img_txt)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # Fallback to default font if the specified one is not found
        font = ImageFont.load_default()
        print(f"Warning: Font {font_path} not found, using default font")
        
    word_index_list = []
    
    for i, tokens in enumerate(sentences):
        # Filter out special tokens for display purposes
        display_tokens = [w for w in tokens if not w.startswith("[")]
        display_tokens = ["[CLS]"] + display_tokens
        
        word_indices = []
        current_word = ""
        
        for j, token in enumerate(tokens):
            # Handle WordPiece tokenization (tokens starting with ##)
            current_word += token.strip("#")
            
            # Check if this is the end of a word or the last token
            is_word_end = (j == len(tokens) - 1) or not tokens[j + 1].startswith("#") if j < len(tokens) - 1 else True
            
            if is_word_end:
                word_indices.append(j)
                # Draw the complete word
                draw.text(
                    ((len(word_indices) - 1 + offset_x) * (vis_size + offset_y), i * FONT_MAX_SIZE),
                    current_word,
                    font=font,
                    fill=(255, 255, 255, 255)
                )
                current_word = ""
        
        word_index_list.append(word_indices)
    
    return img_txt, sentences, word_index_list


def build_attention_images(
        images: torch.Tensor,
        attention_maps: List[torch.Tensor],
        sentences: List[List[str]],
        num_visualizations: int = 8,
        random_selection: bool = False,
        use_rainbow: bool = True,
    ) -> Tuple[Optional[np.ndarray], List[List[str]]]:
    """
    Build a comprehensive visualization of attention maps for multimodal models.
    
    This function creates a visualization that shows how the model attends to different
    parts of an image when processing text tokens. For each input image, the output
    consists of three vertically stacked rows:
    
    1. Text row: Displays the tokens/words of the corresponding text
    2. Raw attention maps row: Shows the unmodified attention values which may appear
       black early in training or when attention values are very low
    3. Merged attention maps row: Shows the attention maps overlaid on the original image,
       making patterns more visible
    
    The first column always shows the original image, the second column shows the maximum
    attention across all tokens, and subsequent columns show attention for individual tokens.
    
    Args:
        images: Tensor of input images [batch_size, channels, height, width]
        attention_maps: List of attention maps from the model, typically from
                        cross-attention layers
        sentences: List of tokenized sentences corresponding to each image
        num_visualizations: Maximum number of examples to include in the visualization
        random_selection: If True, randomly select images; if False, use the first N images
        use_rainbow: If True, use matplotlib's rainbow colormap for attention visualization;
                    if False, use the original grayscale visualization
    """
    # Extract attention map size
    attention_size = attention_maps[0].shape[-1]
    batch_size = images.shape[0]
    
    # Count actual words (excluding special tokens and WordPiece continuations)
    word_counts = []
    for sent in sentences:
        words = [s for s in sent if (not s.startswith("#")) and (not s.startswith("["))]
        word_counts.append(len(words) + 1)  # +1 for [CLS]
    max_word_count = max(word_counts)
    
    # Select which images to visualize
    if random_selection:
        visualization_indices = np.random.choice(len(images), size=min(num_visualizations, len(images)), replace=False)
    else:
        visualization_indices = np.arange(min(num_visualizations, len(images)))
    
    # Determine visualization size
    if attention_size in (17, 19):
        vis_size = attention_size * 16
    else:
        vis_size = images.size(2)
    
    # Create text canvas
    text_canvas = np.ones(
        [batch_size * FONT_MAX_SIZE, (max_word_count + 2) * (vis_size + 2), 3], 
        dtype=np.uint8
    )
    
    # Color-code each word position
    for i in range(max_word_count):
        start_x = (i + 2) * (vis_size + 2)
        end_x = (i + 3) * (vis_size + 2)
        text_canvas[:, start_x:end_x, :] = COLOR_DICT[i]
    
    # Resize images to visualization size
    resized_images = F.interpolate(images, size=(vis_size, vis_size), mode="bilinear", align_corners=False)
    
    # Convert images from [-1, 1] to [0, 255] for visualization
    resized_images = ((resized_images + 1) / 2 * 255).clamp(0, 255).cpu().numpy()
    resized_images = np.transpose(resized_images, (0, 2, 3, 1))
    
    # Create padding elements
    pad_shape = resized_images.shape
    middle_pad = np.zeros([pad_shape[2], 2, 3])
    post_pad = np.zeros([pad_shape[1], pad_shape[2], 3])
    
    # Draw captions and get word indices
    text_map, processed_sentences, word_index_list = draw_caption(text_canvas, vis_size, sentences)
    text_map = np.asarray(text_map).astype(np.uint8)

    rainbow_cmap = cm.get_cmap('rainbow') # 🛠️

    img_set = []
    for idx in visualization_indices:
        # Get and process attention maps for this image
        attn = attention_maps[idx].cpu().view(1, -1, attention_size, attention_size)
        
        # Add maximum attention across all words
        attn_max, _ = attn.max(dim=1, keepdim=True)
        attn = torch.cat([attn_max, attn], 1)
        
        # Reshape and repeat for visualization
        attn = attn.view(-1, 1, attention_size, attention_size).repeat(1, 3, 1, 1).detach().numpy()
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn_maps = attn.shape[0]
        
        # Get the image
        img = resized_images[idx]
        
        # Initialize visualization rows
        row = [img, middle_pad]
        row_merge = [img, middle_pad]
        row_before_norm = []
        
        # Track global min/max for normalization
        min_val_global, max_val_global = 1, 0
        
        # Process word-level attention
        word_end_indices = [0] + [idx + 1 for idx in word_index_list[idx]]
        word_level_attn = []
        
        for j in range(num_attn_maps):
            attn_map = attn[j]
            
            # Upscale attention map if needed
            if (vis_size // attention_size) > 1:
                attn_map = skimage.transform.pyramid_expand(
                    attn_map, sigma=20, upscale=vis_size // attention_size, channel_axis=2
                )
            
            word_level_attn.append(attn_map)
            
            # Process only at word boundaries
            if j in word_end_indices:
                attn_map = np.mean(word_level_attn, axis=0)
                word_level_attn = []
            else:
                continue
            
            row_before_norm.append(attn_map)
            
            # Update global min/max
            min_val = attn_map.min()
            max_val = attn_map.max()
            min_val_global = min(min_val_global, min_val)
            max_val_global = max(max_val_global, max_val)
        
        # Process each word position
        for j in range(max_word_count + 1):
            if j < len(row_before_norm):
                # Normalize and scale to [0, 255]
                attn_map = row_before_norm[j]
                normalized_map = (attn_map - min_val_global) / (max_val_global - min_val_global + 1e-8) 

                if use_rainbow:
                    # Apply rainbow colormap to the normalized values
                    grayscale = np.mean(normalized_map, axis=2)  # Use a weighted average
                    colored_map = rainbow_cmap(grayscale)[:, :, :3]
                    # Convert to uint8 for visualization
                    colored_map = (colored_map * 255).astype(np.uint8)

                # Create attention overlay
                img_pil = Image.fromarray(np.uint8(img))
                attn_pil = Image.fromarray(np.uint8(normalized_map * 255))
                merged = Image.new("RGBA", (vis_size, vis_size), (0, 0, 0, 0))
                mask = Image.new("L", (vis_size, vis_size), 210)  # Semi-transparent mask
                
                merged.paste(img_pil, (0, 0))
                merged.paste(attn_pil, (0, 0), mask)
                merged = np.array(merged)[:, :, :3]
            else:
                attn_map = post_pad
                merged = post_pad

            if use_rainbow:
                row.append(colored_map)
            else:
                row.append(attn_map)

            row.append(middle_pad)
            row_merge.append(merged)
            row_merge.append(middle_pad)

        row = np.concatenate(row, 1)
        row_merge = np.concatenate(row_merge, 1)
        txt = text_map[idx * FONT_MAX_SIZE : (idx + 1) * FONT_MAX_SIZE]
        
        # Check if dimensions match
        if txt.shape[1] != row.shape[1]:
            print(f"Dimension mismatch: txt {txt.shape}, row {row.shape}")
            return None, processed_sentences
        
        # Combine text and visualization
        row = np.concatenate([txt, row, row_merge], 0)
        img_set.append(row)
    
    # Combine all examples
    try:
        combined_img = np.concatenate(img_set, 0)
        return combined_img.astype(np.uint8), processed_sentences
    except ValueError:
        print("Failed to concatenate visualization images")
        return None, processed_sentences
    

def plot_attn_maps(config, attn_maps, imgs, sents, epoch_idx=0, batch_idx=0):

    img_set, _ = build_attention_images(
        imgs,
        attn_maps,
        num_visualizations=config.misc.nvis,
        random_selection=config.misc.rand_vis,
        sentences=sents,
    )

    if img_set is not None:
        im = Image.fromarray(img_set)
        fullpath = (
            f"{config.output_dir}/"
            f"attention_maps_epoch{epoch_idx}_"
            f"{batch_idx}.png"
        )
        im.save(fullpath)
