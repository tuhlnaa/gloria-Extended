from .gloria import GLoRIAModel
from .classification import ClassificationModel

PYTORCH_MODULES = {
    "pretrain": GLoRIAModel,
    "classification": ClassificationModel,
    #"segmentation": SegmentationModel,
}
