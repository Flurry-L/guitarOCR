"""Explicit label/span pointers followed by learned event-to-span coverage.

One encoder pass serves every event in a complete current/previous system pair.
No onset-to-pixel interpolation or fixed line-membership decision is used.
"""

import torch
from torch import nn
from torch.nn import functional as F
from research.models.technique_event_grounding import image_encoder


class ImageQueryLayer(nn.Module):
    """Read shared image features without communicating between event queries."""

    def __init__(self, channels):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)
        self.linear1 = nn.Linear(channels, channels * 2)
        self.linear2 = nn.Linear(channels * 2, channels)

    def forward(self, query, memory):
        attended = self.multihead_attn(query, memory, memory, need_weights=False)[0]
        query = self.norm2(query + attended)
        return self.norm3(query + self.linear2(F.gelu(self.linear1(query))))


class TechniqueSpanRelationNetwork(nn.Module):
    def __init__(self, channels=96):
        super().__init__()
        self.features = image_encoder(channels)
        self.position = nn.Sequential(
            nn.Linear(2, channels), nn.SiLU(), nn.Linear(channels, channels)
        )
        self.query = nn.Sequential(nn.Linear(channels + 2, channels), nn.SiLU())
        self.effect = nn.Parameter(torch.randn(2, channels) * 0.02)
        self.layers = nn.ModuleList([ImageQueryLayer(channels) for _ in range(2)])
        self.pointer_roles = nn.Parameter(torch.randn(3, channels) * 0.02)
        self.pointer_query = nn.Linear(channels, channels)
        self.pointer_key = nn.Linear(channels, channels)
        self.end_query = nn.Linear(channels * 3, channels)
        self.end_relation = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.end_query.weight)
        nn.init.zeros_(self.end_query.bias)
        nn.init.zeros_(self.end_relation[-1].weight)
        nn.init.zeros_(self.end_relation[-1].bias)
        self.event_pointer = nn.Linear(channels, channels)
        self.pointer_offset = nn.Sequential(
            nn.Linear(channels * 2, 16), nn.SiLU(), nn.Linear(16, 2)
        )
        nn.init.zeros_(self.pointer_offset[-1].weight)
        nn.init.zeros_(self.pointer_offset[-1].bias)
        self.anchor_projection = nn.Linear(channels, channels)
        nn.init.zeros_(self.anchor_projection.weight)
        nn.init.zeros_(self.anchor_projection.bias)
        self.pointer_relation = nn.Sequential(
            nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 16)
        )
        self.pointer_relation_query = nn.Linear(channels, 16)
        nn.init.zeros_(self.pointer_relation[-1].weight)
        nn.init.zeros_(self.pointer_relation[-1].bias)
        self.anchor_size = nn.Linear(channels, 2)
        nn.init.constant_(self.anchor_size.bias, -3.5)
        self.presence = nn.Linear(channels, 1)
        self.label_box = nn.Sequential(
            nn.Linear(40 * 5 * 13, channels), nn.SiLU(), nn.Linear(channels, 4)
        )
        nn.init.zeros_(self.label_box[-1].weight)
        nn.init.zeros_(self.label_box[-1].bias)
        self.glyph_presence = nn.Sequential(
            nn.Linear(40 * 3 * 9, channels), nn.SiLU(), nn.Linear(channels, 2)
        )
        self.glyph_fusion = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.glyph_fusion[-1].weight)
        nn.init.zeros_(self.glyph_fusion[-1].bias)
        self.register_buffer("score_fusion_marker", torch.tensor(1))
        self.unknown = nn.Linear(channels, 1)
        self.coverage = nn.Sequential(
            nn.Linear(12, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU(), nn.Linear(32, 1)
        )

    @classmethod
    def from_state_dict(cls, weights):
        """Load the current network strictly; incompatible checkpoints are rejected."""
        model = cls(channels=weights["presence.weight"].shape[1])
        model.load_state_dict(weights)
        return model

    def coverage_from_geometry(self, images, events, geometry):
        span, anchor = (geometry[..., :4], geometry[..., 4:])
        point = events[:, :, None].expand(-1, -1, 2, -1)
        scale = images.new_tensor([120.0, 64.0])
        relative = torch.cat(
            (
                (point - span[..., :2]) * scale,
                (span[..., 2:] - point) * scale,
                (point - (anchor[..., :2] + anchor[..., 2:]) / 2) * scale,
                (span[..., 2:] - span[..., :2]) * scale,
                (anchor[..., 2:] - anchor[..., :2]) * scale,
                (anchor[..., :2] - span[..., :2]) * scale,
            ),
            -1,
        )
        return self.coverage(relative).squeeze(-1)

    def forward(self, images, events, padding):
        detail = self.features[:12](images)
        feature = self.features[12:](detail)
        batch, channels, height, width = feature.shape
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, height, device=images.device),
            torch.linspace(0, 1, width, device=images.device),
            indexing="ij",
        )
        xy = torch.stack((xx, yy), -1).reshape(1, height * width, 2)
        memory = feature.flatten(2).transpose(1, 2) + self.position(xy)
        local = (
            F.grid_sample(feature, events[:, :, None] * 2 - 1, align_corners=True)
            .squeeze(-1)
            .transpose(1, 2)
        )
        query = self.query(torch.cat((local, events), -1))[:, :, None] + self.effect
        count = events.shape[1]
        query = query.reshape(batch, count * 2, channels)
        for layer in self.layers:
            query = layer(query, memory)
        query = query.reshape(batch, count, 2, channels)
        pointer_query = self.pointer_query(query[..., None, :] + self.pointer_roles)
        pointer_logits = (
            torch.bmm(
                pointer_query.reshape(batch, count * 6, channels),
                self.pointer_key(memory).transpose(1, 2),
            )
            / channels**0.5
        )
        relative = (xy[:, None] - events[:, :, None]) * images.new_tensor([120.0, 64.0])
        relation = self.pointer_relation(relative)
        roles = torch.cat(
            (
                pointer_query.reshape(batch, count, 6, channels),
                self.event_pointer(query.mean(2))[:, :, None],
            ),
            2,
        )
        relation = (
            torch.matmul(self.pointer_relation_query(roles), relation.transpose(-1, -2))
            / 4
        )
        pointer_logits = pointer_logits + relation[:, :, :6].reshape(
            batch, count * 6, height * width
        )
        roles = pointer_logits.reshape(batch, count, 2, 3, height * width)
        start_peak, label_peak = (
            roles[..., 0, :].argmax(-1),
            roles[..., 2, :].argmax(-1),
        )

        def selected_feature(indices):
            return torch.gather(
                memory, 1, indices.reshape(batch, count * 2, 1).expand(-1, -1, channels)
            ).reshape(batch, count, 2, channels)

        linked_query = self.end_query(
            torch.cat(
                (query, selected_feature(start_peak), selected_feature(label_peak)), -1
            )
        )
        visual = (
            torch.bmm(
                linked_query.reshape(batch, count * 2, channels),
                self.pointer_key(memory).transpose(1, 2),
            ).reshape(batch, count, 2, height * width)
            / channels**0.5
        )
        start, label = (xy[0][start_peak], xy[0][label_peak])
        relative = torch.cat(
            (
                xy[:, None, None] - start[..., None, :],
                xy[:, None, None] - label[..., None, :],
            ),
            -1,
        ) * images.new_tensor([120.0, 64.0, 120.0, 64.0])
        endpoint = roles[..., 1, :] + visual + self.end_relation(relative).squeeze(-1)
        pointer_logits = torch.stack(
            (roles[..., 0, :], endpoint, roles[..., 2, :]), -2
        ).reshape(batch, count * 6, height * width)
        pointer_attention = pointer_logits.softmax(-1)
        soft_locations = torch.bmm(pointer_attention, xy.expand(batch, -1, -1))
        peaks = pointer_logits.argmax(-1)
        selected = torch.gather(memory, 1, peaks[..., None].expand(-1, -1, channels))
        offset = self.pointer_offset(
            torch.cat((pointer_query.reshape(batch, count * 6, channels), selected), -1)
        )
        locations = xy[0][peaks] + offset.tanh() / images.new_tensor(
            [width - 1.0, height - 1.0]
        )
        if self.training:
            locations = locations + soft_locations - soft_locations.detach()
        locations = locations.reshape(batch, count, 2, 3, 2)
        extent = self.anchor_size(query).sigmoid()
        anchor_center = locations[..., 2, :]
        gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, 5, device=images.device),
            torch.linspace(-1, 1, 13, device=images.device),
            indexing="ij",
        )
        grid = anchor_center[..., None, None, :] + extent[
            ..., None, None, :
        ] * torch.stack((gx, gy), -1)
        region = F.grid_sample(
            detail,
            grid.reshape(batch, count * 2 * 5, 13, 2) * 2 - 1,
            align_corners=True,
            padding_mode="zeros",
        )
        region = region.reshape(batch, 40, count, 2, 5, 13).permute(0, 2, 3, 1, 4, 5)
        delta = self.label_box(region.reshape(batch, count, 2, 40 * 5 * 13))
        anchor_center = anchor_center + delta[..., :2].tanh() * extent
        extent = extent * delta[..., 2:].clamp(-2, 2).exp()
        geometry = torch.cat(
            (
                locations[..., 0, :],
                locations[..., 1, :],
                anchor_center - extent / 2,
                anchor_center + extent / 2,
            ),
            -1,
        )
        event_pointer_logits = (
            torch.bmm(
                self.event_pointer(query.mean(2)),
                self.pointer_key(memory).transpose(1, 2),
            )
            / channels**0.5
        )
        event_pointer_logits = event_pointer_logits + relation[:, :, 6]
        refined_events = torch.bmm(
            event_pointer_logits.softmax(-1), xy.expand(batch, -1, -1)
        )
        coverage_logits = self.coverage_from_geometry(images, refined_events, geometry)
        anchor_feature = (
            F.grid_sample(
                feature,
                anchor_center.reshape(batch, count * 2, 1, 2) * 2 - 1,
                align_corners=True,
            )
            .squeeze(-1)
            .transpose(1, 2)
        )
        presence_query = query + self.anchor_projection(
            anchor_feature.reshape(batch, count, 2, channels)
        )
        gy, gx = torch.meshgrid(
            torch.linspace(-0.5, 0.5, 3, device=images.device),
            torch.linspace(-0.5, 0.5, 9, device=images.device),
            indexing="ij",
        )
        grid = anchor_center[..., None, None, :] + extent[
            ..., None, None, :
        ] * torch.stack((gx, gy), -1)
        glyphs = F.grid_sample(
            detail,
            grid.reshape(batch, count * 2 * 3, 9, 2) * 2 - 1,
            align_corners=True,
            padding_mode="zeros",
        )
        glyphs = glyphs.reshape(batch, 40, count, 2, 3, 9).permute(0, 2, 3, 1, 4, 5)
        word_logits = self.glyph_presence(glyphs.reshape(batch, count, 2, 40 * 3 * 9))
        presence_logits = torch.stack(
            (word_logits[:, :, 0, 0], word_logits[:, :, 1, 1]), -1
        )
        context_logits = self.presence(presence_query)
        other = torch.stack((word_logits[:, :, 0, 1], word_logits[:, :, 1, 0]), -1)
        scores = torch.stack((context_logits.squeeze(-1), presence_logits, other), -1)
        presence_logits = presence_logits + self.glyph_fusion(scores).squeeze(-1)
        positive = presence_logits.sigmoid() * coverage_logits.sigmoid()
        pm, let = (positive[..., 0], positive[..., 1])
        unknown = self.unknown(query.mean(2)).squeeze(-1).sigmoid()
        probability = torch.stack(
            ((1 - pm) * (1 - let), pm * (1 - let), let * (1 - pm), pm * let), -1
        ) * (1 - unknown[..., None])
        probability = torch.cat((probability, unknown[..., None]), -1)
        unused = torch.zeros_like(probability)
        unused[..., -1] = 1
        probability = torch.where(padding[..., None], unused, probability)
        result = (
            probability.clamp_min(1e-07).log(),
            geometry,
            presence_logits,
            coverage_logits,
            pointer_logits.reshape(batch, count, 2, 3, height * width),
            event_pointer_logits,
            refined_events,
        )
        return result
