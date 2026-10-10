"""Small visual classifier for relations between adjacent detected staff rows."""
from __future__ import annotations

import torch
from torch import nn


class RelationNet(nn.Module):
    """CNN retains horizontal position so names and connectors remain visible."""
    def __init__(self):
        super().__init__()
        layers = []
        channels = [1, 16, 32, 48, 64]
        for a, b in zip(channels, channels[1:]):
            layers += [nn.Conv2d(a, b, 3, stride=2, padding=1), nn.BatchNorm2d(b), nn.SiLU(),
                       nn.Conv2d(b, b, 3, padding=1, groups=b), nn.BatchNorm2d(b), nn.SiLU()]
        self.image = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((4, 16)), nn.Flatten(), nn.Linear(64*4*16, 128), nn.SiLU())
        self.geometry = nn.Sequential(nn.Linear(12, 32), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(160, 64), nn.SiLU(), nn.Dropout(.1), nn.Linear(64, 1))

    def forward(self, pixels, geometry):
        visual = self.image(1-pixels)
        spatial = self.geometry(geometry)
        return self.head(torch.cat((visual, spatial), dim=1)).squeeze(1)


class PartRelationNet(RelationNet):
    """Shared image features with a second conditional same-part classifier."""
    def __init__(self):
        super().__init__()
        self.part_head = nn.Sequential(nn.Linear(160, 64), nn.SiLU(), nn.Dropout(.1), nn.Linear(64, 1))

    def forward(self, pixels, geometry):
        features = torch.cat((self.image(1-pixels), self.geometry(geometry)), dim=1)
        return self.head(features).squeeze(1), self.part_head(features).squeeze(1)


class StructureRelationNetwork(nn.Module):
    """The adopted independent system and same-part image branches."""

    def __init__(self, system, part):
        super().__init__()
        self.system = system
        self.part_image = part.image
        self.part_geometry = part.geometry
        self.part_head = part.part_head

    def forward(self, pixels, geometry):
        features = torch.cat((self.part_image(1-pixels), self.part_geometry(geometry)), dim=1)
        return self.system(pixels, geometry), self.part_head(features).squeeze(1)
