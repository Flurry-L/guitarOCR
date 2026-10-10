"""Auxiliary readers used by the current recognition pipeline."""

from torch import nn
from research.common.clef_spatial import LABELS
import torch
from torch.nn import functional as F
from research.common.signature_image import SPATIAL_CONFIG
from research.common.signature_image import validate_signature_config


class ChordTextRoles(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        c = 4
        for n in (24, 48, 80, 112):
            layers.extend(
                [
                    nn.Conv2d(c, n, 3, stride=2, padding=1),
                    nn.BatchNorm2d(n),
                    nn.GELU(),
                    nn.Conv2d(n, n, 3, padding=1, groups=n),
                    nn.BatchNorm2d(n),
                    nn.GELU(),
                ]
            )
            c = n
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((8, 8)))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(c * 64, 64), nn.GELU(), nn.Linear(64, 2)
        )

    def forward(self, images):
        return self.head(self.features(images))


class ClefSpatialNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for index, (source, target) in enumerate(zip([2, 16, 24, 40], [16, 24, 40, 64])):
            layers.extend([nn.Conv2d(source, target, 3, stride=1 if index == 0 else 2, padding=1),
                           nn.BatchNorm2d(target), nn.SiLU(),
                           nn.Conv2d(target, target, 3, padding=1, groups=target),
                           nn.BatchNorm2d(target), nn.SiLU()])
        self.features = nn.Sequential(*layers, nn.Conv2d(64, 16, 1), nn.SiLU(), nn.AvgPool2d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16 * 16 * 16, 128), nn.SiLU(),
                                  nn.Dropout(.1), nn.Linear(128, len(LABELS)))

    def forward(self, images):
        return self.head(self.features(torch.cat([1 - images[:, :1], images[:, 1:]], dim=1)))


class InstrumentTextNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(257, 24, padding_idx=0)
        self.features = nn.ModuleList(
            [nn.Conv1d(24, 32, k, padding=k // 2) for k in (3, 5, 7)]
        )
        self.head = nn.Sequential(nn.Linear(96, 64), nn.GELU(), nn.Linear(64, 5))

    def forward(self, tokens):
        embedded = self.embedding(tokens).transpose(1, 2)
        features = torch.cat(
            [F.gelu(layer(embedded)).amax(2) for layer in self.features], dim=1
        )
        return self.head(features)


def make_model(pretrained=False, *, proposal_width=12):
    import torch
    from torch import nn
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    pool_shape = (2, 6)

    class QualityHead(nn.Module):
        def __init__(self):
            super().__init__()
            base = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            )
            original = base.features[0][0]
            first = nn.Conv2d(4, 16, 3, stride=2, padding=1, bias=False)
            with torch.no_grad():
                first.weight[:, :3].copy_(original.weight)
                first.weight[:, 3:].zero_()
            base.features[0][0] = first
            self.features = base.features
            self.pool = nn.AdaptiveAvgPool2d(pool_shape)
            self.head = nn.Sequential(
                nn.Linear(576 * pool_shape[0] * pool_shape[1] + proposal_width, 128),
                nn.Hardswish(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
            )
            self.register_buffer(
                "mean", torch.tensor([0.485, 0.456, 0.406, 0.5])[None, :, None, None]
            )
            self.register_buffer(
                "std", torch.tensor([0.229, 0.224, 0.225, 0.5])[None, :, None, None]
            )

        def logits(self, image, proposal):
            x = (image - self.mean) / self.std
            x = self.features(x)
            x = self.pool(x).flatten(1)
            logits = self.head(torch.cat([x, proposal], 1)).squeeze(1)
            return logits

        def forward(self, image, proposal):
            return self.logits(image, proposal).sigmoid()

    return QualityHead()


def encoder(channels, pool, output):
    layers = []
    for source, target in zip(channels, channels[1:]):
        layers.extend([nn.Conv2d(source, target, 3, stride=2, padding=1),
                       nn.BatchNorm2d(target), nn.SiLU(),
                       nn.Conv2d(target, target, 3, padding=1, groups=target),
                       nn.BatchNorm2d(target), nn.SiLU()])
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(pool), nn.Flatten(),
                         nn.Linear(channels[-1] * pool[0] * pool[1], output), nn.SiLU())


class PartContinuationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.row = encoder([1, 16, 24, 32, 48], (2, 12), 96)
        self.context = encoder([2, 8, 16, 24, 32], (4, 4), 64)
        self.head = nn.Sequential(nn.Linear(96 * 4 + 64 * 2 + 17, 128), nn.SiLU(),
                                  nn.Dropout(0.1), nn.Linear(128, 3))

    def forward(self, rows, contexts, features):
        before = self.row(1 - rows[:, :1])
        after = self.row(1 - rows[:, 1:])
        left = self.context(torch.cat([1 - contexts[:, :1], contexts[:, 1:2]], 1))
        right = self.context(torch.cat([1 - contexts[:, 2:3], contexts[:, 3:4]], 1))
        return self.head(torch.cat([before, after, (before - after).abs(), before * after,
                                    left, right, features], 1))


def block(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1, groups=cout, bias=False),
        nn.BatchNorm2d(cout),
        nn.GELU(),
    )


class StaffLabelNetwork(nn.Module):
    """Dense learned text-role queries retain the entire input staff context."""

    def __init__(self):
        super().__init__()
        self.stem = block(3, 24, 2)
        self.c2 = block(24, 48, 2)
        self.c3 = block(48, 80, 2)
        self.c4 = block(80, 112, 2)
        self.lateral2 = nn.Conv2d(48, 48, 1)
        self.lateral3 = nn.Conv2d(80, 48, 1)
        self.lateral4 = nn.Conv2d(112, 48, 1)
        self.context = nn.Conv2d(112, 48, 1)
        self.head = nn.Sequential(block(48, 48), nn.Conv2d(48, 9, 1))
        nn.init.constant_(self.head[-1].bias[0:1], -4.0)
        nn.init.constant_(self.head[-1].bias[1:5], -3.0)

    def forward(self, images):
        c2 = self.c2(self.stem(images))
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        shape = c2.shape[-2:]
        feature = self.lateral2(c2)
        feature = feature + F.interpolate(
            self.lateral3(c3), size=shape, mode="bilinear", align_corners=False
        )
        feature = feature + F.interpolate(
            self.lateral4(c4), size=shape, mode="bilinear", align_corners=False
        )
        feature = feature + self.context(F.adaptive_avg_pool2d(c4, 1))
        output = self.head(feature)
        return output[:, :1], output[:, 1:5].sigmoid(), output[:, 5:]


class StaffNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        incoming = 3
        for channels, stride in [(24, 2), (40, 2), (64, (1, 2)), (64, (1, 2))]:
            layers.extend(
                [
                    nn.Conv2d(incoming, channels, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(),
                    nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(),
                ]
            )
            incoming = channels
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((32, 4)),
            nn.Flatten(),
            nn.Linear(8192, 128),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 9),
        )

    def forward(self, images):
        return self.head(self.features(images))


class SpatialSignatureNetwork(nn.Module):
    """Learn where signatures occur in the complete image, without staff cropping.

    Independent key/time queries keep small signature evidence separate from
    globally pooled context. Coordinate channels describe position but impose
    no left-prefix mask. The dilated backbone retains an output stride of 16.
    """

    def __init__(self, pretrained=False):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        base = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None,
            dilated=True,
        )
        self.features = base.features
        self.project = nn.Sequential(
            nn.Conv2d(576, 128, 1), nn.BatchNorm2d(128), nn.Hardswish()
        )
        self.attention = nn.Sequential(
            nn.Conv2d(130, 64, 1), nn.Hardswish(), nn.Conv2d(64, 2, 1)
        )
        self.key_head = nn.Sequential(
            nn.Linear(384, 128), nn.Hardswish(), nn.Dropout(0.15), nn.Linear(128, 16)
        )
        self.time_head = nn.Sequential(
            nn.Linear(384, 128), nn.Hardswish(), nn.Dropout(0.15), nn.Linear(128, 41)
        )

    def visual_features(self, images):
        values = self.project(self.features(images))
        batch, _, height, width = values.shape
        x = (
            torch.linspace(-1, 1, width, device=values.device, dtype=values.dtype)
            .view(1, 1, 1, width)
            .expand(batch, 1, height, width)
        )
        y = (
            torch.linspace(-1, 1, height, device=values.device, dtype=values.dtype)
            .view(1, 1, height, 1)
            .expand(batch, 1, height, width)
        )
        return values, torch.cat((values, x, y), 1)

    def signature_vectors(self, values, positioned):
        weights = self.attention(positioned).flatten(2).softmax(-1)
        flat = values.flatten(2)
        focused = torch.bmm(weights, flat.transpose(1, 2))
        global_context = torch.cat((flat.mean(-1), flat.amax(-1)), -1)
        return torch.cat((focused[:, 0], global_context), -1), torch.cat(
            (focused[:, 1], global_context), -1
        )

    def forward(self, images):
        values, positioned = self.visual_features(images)
        key_vector, time_vector = self.signature_vectors(values, positioned)
        key = self.key_head(key_vector)
        time = self.time_head(time_vector)
        return key, time[:, :33], time[:, 33:]


def make_signature_network(config=None, pretrained=False):
    validate_signature_config(SPATIAL_CONFIG if config is None else config)
    return SpatialSignatureNetwork(pretrained)
