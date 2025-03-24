import sys
import torch

from pathlib import Path
from torchinfo import summary
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from gloria.models.pytorch.classification import ClassificationModel

config = OmegaConf.load("./test/classification_config.yaml")

# Create DenseNet model instance
model = ClassificationModel(config)

# Test with random input (DenseNet expects 3 channel input)
x = torch.randn(1, 3, 224, 224)  # Standard ImageNet input size
output = model(x)

# Print model architecture
print(model)

# Print output shape
print("\nOutput shape:", output.shape)

# Print detailed model summary
summary(model, input_data=(x,))

"""
Used PretrainedImageClassifier

Traceback (most recent call last):
  File "e:\Kai_2\CODE_Repository\gloria-Extended\test\print_ClassificationModel.py", line 20, in <module>
    output = model(x)
             ^^^^^^^^
TypeError: 'ClassificationModel' object is not callable
"""