import torch.nn as nn
from torchvision import models as models_2d


class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""

    def forward(self, x):
        return x


################################################################################
# ResNet Family
################################################################################


def resnet_18(pretrained='DEFAULT'):
    model = models_2d.resnet18(weights=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


def resnet_34(pretrained='DEFAULT'):
    model = models_2d.resnet34(weights=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


def resnet_50(pretrained='DEFAULT'):
    model = models_2d.resnet50(weights=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


################################################################################
# DenseNet Family
################################################################################


def densenet_121(pretrained='DEFAULT'):
    model = models_2d.densenet121(weights=pretrained)
    feature_dims = model.classifier.in_features
    model.classifier = Identity()
    return model, feature_dims, None


def densenet_161(pretrained='DEFAULT'):
    model = models_2d.densenet161(weights=pretrained)
    feature_dims = model.classifier.in_features
    model.classifier = Identity()
    return model, feature_dims, None


def densenet_169(pretrained='DEFAULT'):
    model = models_2d.densenet169(weights=pretrained)
    feature_dims = model.classifier.in_features
    model.classifier = Identity()
    return model, feature_dims, None


################################################################################
# ResNextNet Family
################################################################################


def resnext_50(pretrained='DEFAULT'):
    model = models_2d.resnext50_32x4d(weights=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, None


def resnext_100(pretrained='DEFAULT'):
    model = models_2d.resnext101_32x8d(weights=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, None
