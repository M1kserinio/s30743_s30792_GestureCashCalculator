import torch.nn as nn


def make_resnet18(class_count, pretrained):
    """ResNet-18 z ostatnią warstwą podmienioną na liczbę nominałów."""
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, class_count)
    return model
