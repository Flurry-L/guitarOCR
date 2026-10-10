"""Small spatial reader for printed diagram evidence, independent of chord names."""

import math

import torch
from torch import nn

SYMBOLS = '_0123456789IVX'
MAX_TEXT_LENGTH = 6


def roman(number):
    return 'X' * (number // 10) + ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX'][number % 10]


def tokens(text):
    return [SYMBOLS.index(character) for character in text]


class NumeralCTC(nn.Module):
    """Exact CTC likelihoods for the two printed forms of values 0 through 36.

    The grammar converts recognized characters into a fret number. Its states
    never inspect pixels, select a crop or prescribe a font's glyph geometry.
    """
    def __init__(self):
        super().__init__()
        sequences=[tokens(text) for number in range(37) for text in (str(number),roman(number) or str(number))]
        width=2*max(map(len,sequences))+1
        labels=torch.zeros(len(sequences),width,dtype=torch.long)
        ends=[]
        for row,sequence in enumerate(sequences):
            labels[row,1:2*len(sequence):2]=torch.tensor(sequence)
            ends.append([2*len(sequence)-1,2*len(sequence)])
        skip=torch.zeros_like(labels,dtype=torch.bool)
        skip[:,2:]=(labels[:,2:]!=0)&(labels[:,2:]!=labels[:,:-2])
        self.register_buffer('labels',labels)
        self.register_buffer('skip',skip)
        self.register_buffer('ends',torch.tensor(ends,dtype=torch.long))

    def forward(self,characters):
        probability=characters.float().log_softmax(-1)
        batch=characters.shape[0]
        states=probability.new_full((batch,*self.labels.shape),-10000.)
        states[:,:,0]=0
        blocked=states.new_full((batch,self.labels.shape[0],1),-10000.)
        for frame in range(characters.shape[1]):
            step=torch.cat([blocked,states[:,:,:-1]],-1)
            skip=torch.cat([blocked.expand(-1,-1,2),states[:,:,:-2]],-1)
            skip=skip.masked_fill(~self.skip[None],-10000.)
            emission=probability[:,frame][:,self.labels]
            states=torch.logsumexp(torch.stack([states,step,skip]),0)+emission
        final=states.gather(-1,self.ends[None].expand(batch,-1,-1)).logsumexp(-1)
        return final.reshape(batch,37,2).logsumexp(-1)-math.log(2)


OUTPUTS = ("kind", "strings", "base", "frets", "fingers", "barres")


class ChordFingerKeypoints(nn.Module):
    """Pose-style string ownership with a shared local character reader.

    Each string predicts a pixel or the explicit absent class. A small feature
    pyramid supplies the whole grid context without a chord-identity embedding.
    """

    version = 5

    def __init__(self):
        super().__init__()
        self.register_buffer("reader_version", torch.tensor(self.version))
        self.visual = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 32, 3, padding=1), nn.GELU()
        )
        self.character_pixels = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2),
            nn.GELU(),
        )
        self.context = nn.Conv2d(64, 32, 1)
        self.locate = nn.Sequential(
            nn.Conv2d(82, 48, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(48, 32, 3, padding=4, dilation=4),
            nn.GELU(),
            nn.Conv2d(32, 12, 1),
        )
        self.absent = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 12))
        self.read = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 6))
        axis = (torch.arange(64, dtype=torch.float32) + 0.5) / 64
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        coordinates = torch.stack([x.flatten(), y.flatten()], -1)
        waves = coordinates[:, :, None] * torch.tensor([1.0, 2.0, 4.0, 8.0]) * (2 * torch.pi)
        position = torch.cat([coordinates, waves.sin().flatten(1), waves.cos().flatten(1)], -1)
        self.register_buffer("coordinates", coordinates)
        self.register_buffer("position_features", position.T.reshape(1, 18, 64, 64), persistent=False)


    @staticmethod
    def characters_at(features, centers):
        return nn.functional.grid_sample(
            features, centers[:, :, None] * 2 - 1, align_corners=False, padding_mode="border"
        )[:, :, :, 0].transpose(1, 2)

    def forward(self, fine, embedding, coarse, images=None):
        visual = self.visual(fine)
        context = nn.functional.interpolate(
            self.context(coarse), size=(64, 64), mode="bilinear", align_corners=False
        )
        features = torch.cat([visual, context, self.position_features.expand(fine.shape[0], -1, -1, -1)], 1)
        pixels = self.locate(features).flatten(2)
        absent = self.absent(coarse)[..., None] + math.log(64 * 64)
        locations = torch.cat([pixels, absent], -1)
        with torch.autocast(device_type=fine.device.type, enabled=False):
            centers = pixels.float().softmax(-1) @ self.coordinates
        attended = self.characters_at(self.character_pixels(images), centers)
        present = self.read(attended)[..., 1:].log_softmax(-1)
        probability = locations.log_softmax(-1)
        fingers = torch.cat(
            [probability[..., -1:], probability[..., :-1].logsumexp(-1, keepdim=True) + present], -1
        )
        return (fingers, locations)


class ChordBaseReader(nn.Module):
    """Locate printed base text; read its appearance from local features/pixels."""

    def __init__(self):
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 32, 3, padding=1), nn.GELU()
        )
        self.context = nn.Conv2d(64, 32, 1)
        self.locate = nn.Sequential(nn.Conv2d(66, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 1))
        self.absent = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1))
        self.extent = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 2))
        self.glyph = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=(2, 1), padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=(2, 1), padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=(2, 1), padding=1),
            nn.GELU(),
            nn.Identity(),
        )
        self.characters = nn.Conv1d(32, len(SYMBOLS), 1)
        self.numerals = NumeralCTC()
        y, x = torch.meshgrid((torch.arange(32) + 0.5) / 32, (torch.arange(96) + 0.5) / 96, indexing="ij")
        self.register_buffer("region_grid", torch.stack([x, y], -1)[None] * 2 - 1)
        self.role = nn.Sequential(nn.Linear(580, 32), nn.GELU(), nn.Linear(32, 1))
        self.reliability = nn.Sequential(nn.Linear(655, 64), nn.GELU(), nn.Linear(64, 1))
        self.reliability_norm = nn.LayerNorm(580)
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.zeros_(self.reliability[-1].bias)
        self.read_absent = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 37))
        axis = (torch.arange(64, dtype=torch.float32) + 0.5) / 64
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("coordinates", torch.stack([x, y])[None])
        for layer in [] + [self.read_absent[-1]]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def region_span(self, extent):
        aspect = self.region_grid.shape[2] / self.region_grid.shape[1]
        height = torch.maximum(extent[:, 1], extent[:, 0] / aspect) * 1.5
        return torch.stack([height * aspect, height], -1)

    def region_patch(self, images, center, extent):
        grid = self.region_grid
        sampling = (center[:, None, None] + grid * self.region_span(extent)[:, None, None] / 2) * 2 - 1
        return nn.functional.grid_sample(images, sampling, align_corners=False, padding_mode="border")

    def region_features(self, images, center, extent):
        return self.glyph(self.region_patch(images, center, extent))

    def region_characters(self, images, center, extent):
        return self.characters(self.region_features(images, center, extent).mean(-2)).transpose(1, 2)

    def read_region(self, images, center, extent):
        return self.read_features(self.region_features(images, center, extent))

    def read_features(self, features):
        return self.numerals(self.characters(features.mean(-2)).transpose(1, 2))

    def forward(self, fine, coarse, images=None, regions=None, global_base=None, global_embedding=None):
        visual = self.visual(fine)
        context = nn.functional.interpolate(
            self.context(coarse), size=(64, 64), mode="bilinear", align_corners=False
        )
        pixels = self.locate(
            torch.cat([visual, context, self.coordinates.expand(fine.shape[0], -1, -1, -1)], 1)
        ).flatten(1)
        locations = torch.cat([pixels, self.absent(coarse) + math.log(64 * 64)], -1)
        attended = pixels.softmax(-1)[:, None] @ visual.flatten(2).transpose(1, 2)
        features = attended[:, 0]
        extent = self.extent(features).sigmoid()
        with torch.autocast(device_type=fine.device.type, enabled=False):
            center = pixels.float().softmax(-1) @ self.coordinates.flatten(2)[0].T
        role_features = self.region_features(images, center.detach(), extent.detach())
        glyph = role_features.flatten(2)
        role_input = torch.cat(
            [global_embedding, glyph.mean(-1), glyph.amax(-1), center.detach(), extent.detach()], -1
        )
        role = self.role(role_input).float()
        locations = torch.cat(
            [
                pixels.float().log_softmax(-1) + nn.functional.logsigmoid(role),
                nn.functional.logsigmoid(-role),
            ],
            -1,
        )
        reading_extent = extent
        if regions is not None:
            truth_center, truth_extent, visible = regions
            jittered_center = truth_center + (torch.rand_like(truth_center) - 0.5) * truth_extent * 0.2
            jittered_extent = truth_extent * (1 + (torch.rand_like(truth_extent) - 0.5) * 0.2)
            center = torch.where(visible[:, None], jittered_center, center.detach())
            reading_extent = torch.where(visible[:, None], jittered_extent, extent.detach())
        reading = (
            self.read_features(role_features)
            if 8 >= 6 and regions is None
            else self.read_region(images, center, reading_extent)
        )
        presence = locations.float().log_softmax(-1)
        label_presence = presence[:, :-1].logsumexp(-1, keepdim=True)
        absent_reading = global_base
        local_probability = reading.float().log_softmax(-1)
        global_probability = absent_reading.float().log_softmax(-1)
        context = self.reliability_norm(role_input)
        reliability = self.reliability(
            torch.cat([context, global_probability.exp(), local_probability.exp(), label_presence.exp()], -1)
        ).float()
        present_weight = label_presence + nn.functional.logsigmoid(reliability)
        missing_weight = torch.logsumexp(
            torch.stack([presence[:, -1:], label_presence + nn.functional.logsigmoid(-reliability)]), 0
        )
        present = present_weight + local_probability
        missing = missing_weight + global_probability
        logits = torch.logsumexp(torch.stack([present, missing]), 0)
        return (logits, locations, extent)


class ChordEvidenceNetwork(nn.Module):
    def __init__(self, pretrained=False, local_readers=True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.features = nn.Sequential(
            *list(backbone.children())[:7], nn.Conv2d(256, 64, 1), nn.GELU(), nn.AdaptiveAvgPool2d((8, 8))
        )
        self.project = nn.Sequential(nn.Flatten(), nn.Linear(256 * 16, 512), nn.GELU(), nn.Dropout(0.1))
        self.heads = nn.ModuleList((nn.Linear(512, n) for n in [3, 13, 37, 12 * 9, 12 * 6, 7 * 11]))
        if local_readers:
            self.finger_queries = ChordFingerKeypoints()
            self.base_reader = ChordBaseReader()

    @classmethod
    def from_weights(cls, weights):
        """Load either active consensus reader and reject incompatible checkpoints."""
        model = cls(local_readers="base_reader.locate.0.weight" in weights)
        model.load_state_dict(weights)
        return model

    def forward_with_auxiliary(self, images, base_regions=None):
        features = images
        fine = None
        for index, layer in enumerate(self.features):
            features = layer(features)
            if index == 4:
                fine = features
        embedding = self.project(features)
        kind, strings, base, frets, fingers, barres = [head(embedding) for head in self.heads]
        locations = None
        if hasattr(self, "finger_queries"):
            detail, locations = self.finger_queries(fine, embedding, features, images)
            fingers = detail.flatten(1)
        base_detail = base_locations = base_extent = None
        if hasattr(self, "base_reader"):
            base_detail, base_locations, base_extent = self.base_reader(
                fine, features, images, base_regions, base, embedding
            )
            base = base_detail
        outputs = (
            kind,
            strings,
            base,
            frets.reshape(-1, 12, 9),
            fingers.reshape(-1, 12, 6),
            barres.reshape(-1, 7, 11),
        )
        return (outputs, locations, base_detail, base_locations, base_extent)

    def forward(self, images):
        return self.forward_with_auxiliary(images)[0]


class ChordEvidenceConsensus(nn.Module):
    """Two frozen visual readers; only calibrated matching structures auto-apply."""

    def __init__(self, primary, secondary, contracts, agreement_thresholds=None, agreement_fields=None):
        super().__init__()
        self.primary = primary
        self.secondary = secondary
        agreement_thresholds = agreement_thresholds or {"geometry": 0.0, "full": 0.0}
        self.register_buffer(
            "agreement_thresholds", torch.tensor([agreement_thresholds[key] for key in ["geometry", "full"]])
        )
        self.register_buffer(
            "agreement_fields", torch.tensor([(agreement_fields or {}).get(key, 0.0) for key in OUTPUTS])
        )
        for index, contract in enumerate(contracts):
            self.register_buffer(
                f"thresholds_{index}", torch.tensor([contract["thresholds"][key] for key in OUTPUTS])
            )
            self.register_buffer(
                f"temperatures_{index}", torch.tensor([contract["temperature"][key] for key in OUTPUTS])
            )
            self.register_buffer(
                f"joint_{index}",
                torch.tensor([contract["joint_thresholds"][key] for key in ["geometry", "full"]]),
            )

    def accepted(self, outputs, index):
        temperature = getattr(self, f"temperatures_{index}")
        threshold = getattr(self, f"thresholds_{index}")
        probabilities = [torch.softmax(value / temperature[i], -1) for i, value in enumerate(outputs[:5])]
        count = outputs[1].argmax(-1)
        mask = torch.arange(12, device=count.device)[None, :] < count[:, None]
        edges = outputs[5].div(temperature[5]).sigmoid()
        edge_mask = torch.arange(11, device=count.device)[None, None, :] < count[:, None, None] - 1
        confidence = torch.stack(
            [
                *[p.max(-1).values for p in probabilities[:3]],
                *[
                    torch.where(count > 0, p.max(-1).values.masked_fill(~mask, 1.0).amin(-1), 0.0)
                    for p in probabilities[3:]
                ],
                torch.where(
                    count > 1,
                    torch.maximum(edges, 1 - edges).masked_fill(~edge_mask, 1.0).flatten(1).amin(-1),
                    0.0,
                ),
            ],
            -1,
        )
        flags = confidence >= threshold
        geometry = flags[:, [0, 1, 2, 3, 5]].all(-1) & (
            confidence[:, [0, 1, 2, 3, 5]].amin(-1) >= getattr(self, f"joint_{index}")[0]
        )
        full = flags.all(-1) & (confidence.amin(-1) >= getattr(self, f"joint_{index}")[1])
        valid = (outputs[0].argmax(-1) == 0) & (count > 0) & (outputs[2].argmax(-1) > 0)
        return (
            geometry & valid,
            full & valid,
            confidence[:, [0, 1, 2, 3, 5]].amin(-1),
            confidence.amin(-1),
            confidence,
        )

    def forward(self, images):
        primary = self.primary(images)
        secondary = self.secondary(images)
        first = [value.argmax(-1) for value in primary[:5]]
        second = [value.argmax(-1) for value in secondary[:5]]
        mask = torch.arange(12, device=images.device)[None, :] < first[1][:, None]
        edge_mask = torch.arange(11, device=images.device)[None, None, :] < first[1][:, None, None] - 1
        same = (first[0] == second[0]) & (first[1] == second[1]) & (first[2] == second[2])
        same = same & ((first[3] == second[3]) | ~mask).all(-1)
        same = same & (((primary[5] >= 0) == (secondary[5] >= 0)) | ~edge_mask).flatten(1).all(-1)
        a, b = (self.accepted(primary, 0), self.accepted(secondary, 1))
        fields = torch.minimum(a[4], b[4]) >= self.agreement_fields
        geometry = (
            same
            & a[0]
            & b[0]
            & (torch.minimum(a[2], b[2]) >= self.agreement_thresholds[0])
            & fields[:, [0, 1, 2, 3, 5]].all(-1)
        )
        full = (
            geometry
            & a[1]
            & b[1]
            & ((first[4] == second[4]) | ~mask).all(-1)
            & (torch.minimum(a[3], b[3]) >= self.agreement_thresholds[1])
            & fields.all(-1)
        )
        return (*primary, *secondary, torch.stack([geometry.float(), full.float()], -1))
