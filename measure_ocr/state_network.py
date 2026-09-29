"""Torch network used for training and ONNX export."""

from torch import nn


class SignatureNetwork(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        base = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.features = nn.Sequential(*list(base.children())[:-2], nn.AdaptiveAvgPool2d((1, 4)))
        self.head = nn.Sequential(nn.Linear(4096, 512), nn.GELU(), nn.Dropout(.1), nn.Linear(512, 57))

    def forward(self, images):
        features = self.features(images.flatten(0, 1)).reshape(images.shape[0], -1)
        logits = self.head(features)
        return logits[:, :16], logits[:, 16:49], logits[:, 49:]
