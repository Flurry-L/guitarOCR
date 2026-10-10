"""Pixel grounding of known music events, trained from renderer note bounds."""

import torch
from torch import nn


def image_encoder(channels=96):
    """Shared stride-eight encoder for event positions and visible techniques."""
    layers, incoming = ([], 4)
    for outgoing, stride in [(24, 2), (40, 2), (64, 2), (channels, 1)]:
        layers.extend(
            [
                nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False),
                nn.BatchNorm2d(outgoing),
                nn.SiLU(),
                nn.Conv2d(outgoing, outgoing, 3, 1, 1, groups=outgoing, bias=False),
                nn.BatchNorm2d(outgoing),
                nn.SiLU(),
            ]
        )
        incoming = outgoing
    return nn.Sequential(*layers)


class EventGroundingNetwork(nn.Module):
    def __init__(self, feature_dim=32, channels=96):
        super().__init__()
        self.features = image_encoder(channels)
        self.position = nn.Sequential(
            nn.Linear(2, channels), nn.SiLU(), nn.Linear(channels, channels)
        )
        self.music = nn.Sequential(
            nn.Linear(feature_dim, channels), nn.SiLU(), nn.Linear(channels, channels)
        )
        self.decoder = nn.TransformerDecoderLayer(
            channels, 4, channels * 2, dropout=0.0, activation="gelu", batch_first=True
        )
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)

    def forward(self, images, events, padding):
        feature = self.features(images)
        batch, channels, height, width = feature.shape
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, height, device=images.device),
            torch.linspace(0, 1, width, device=images.device),
            indexing="ij",
        )
        xy = (
            torch.stack((xx, yy), -1)
            .reshape(1, height * width, 2)
            .expand(batch, -1, -1)
        )
        memory = feature.flatten(2).transpose(1, 2) + self.position(xy)
        query = self.decoder(self.music(events), memory, tgt_key_padding_mask=padding)
        scores = (
            torch.bmm(self.query(query), self.key(memory).transpose(1, 2))
            / channels**0.5
        )
        attention = scores.softmax(-1)
        positions = torch.bmm(attention, xy)
        return (positions, scores)
