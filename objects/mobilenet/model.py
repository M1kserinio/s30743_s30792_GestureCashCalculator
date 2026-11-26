import torch.nn as nn


def make_mobilenet(class_count, pretrained):
    #MobileNetV3-Small z ostatnia warstwa podmieniona na liczbe klas Lekka siec do telefonow wiec liczy kilka razy mniej niz ResNet-18.

    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, class_count)
    return model
