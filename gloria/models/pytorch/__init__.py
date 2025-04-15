from .segmentation import SegmentationModel
from .gloria import GLoRIAModel
from .classification import ClassificationModel

PYTORCH_MODULES = {
    "pretrain": GLoRIAModel,
    "classification": ClassificationModel,
    "gloria_classification": ClassificationModel,
    "segmentation": SegmentationModel,
}
