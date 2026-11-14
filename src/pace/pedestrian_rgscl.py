"""Three-view region encoder with an independent pedestrian network view."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

VIEW_NAMES = ("poi", "building", "walk")
WALK_FEATURES = ("log1p_length_m_div_5", "cos_2theta", "sin_2theta")


def walk_graph(features, pairs):
    """A node is a walk segment; a graph edge means a shared endpoint."""
    x = torch.as_tensor(features, dtype=torch.float32).reshape(-1, len(WALK_FEATURES))
    pairs = set(pairs)
    bidirectional = sorted(pairs | {(j, i) for i, j in pairs})
    edges = torch.tensor(bidirectional, dtype=torch.long).reshape(-1, 2).T.contiguous()
    return Data(x=x, edge_index=edges)


class ViewAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.scorers = nn.ModuleList(
            nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1, bias=False))
            for _ in VIEW_NAMES
        )

    def forward(self, views, available):
        scores = torch.cat([score(view) for score, view in zip(self.scorers, views, strict=True)], dim=-1)
        if not available.any(dim=-1).all():
            raise ValueError("Every region must have at least one observed view")
        weights = scores.masked_fill(~available, -torch.inf).softmax(dim=-1)
        return (weights[..., None] * torch.stack(views, dim=1)).sum(dim=1), weights


def contrastive_loss(logits, positive):
    pos, neg = logits[positive], logits[~positive]
    if not pos.numel() or not neg.numel():
        return logits.sum() * 0
    return F.softplus(-pos).mean() + F.softplus(neg).mean()


def local_to_region(nodes, source, other_nodes, target, score, block_ids):
    local = ~source.dummy
    valid_regions = target.ptr.diff() > 1
    # Do not train on a missing view's dummy as if it were a region observation.
    local &= valid_regions[source.batch]
    summaries = other_nodes[target.dummy][valid_regions]
    logits = nodes[local] @ score @ summaries.T
    return contrastive_loss(logits, source.batch[local, None] == block_ids[valid_regions])


class ThreeViewRGSCL(nn.Module):
    """Reuse the RGSCL encoder and contrast objectives with a third readout."""

    def __init__(self, encoder, poi_dim, width=64, heads=4):
        super().__init__()
        self.width = width
        self.encoders = nn.ModuleList(
            encoder(dim, width, 1, "gat", heads, "prelu")
            for dim in (poi_dim, 8, len(WALK_FEATURES))
        )
        self.fusion = ViewAttention(width)
        self.norm = nn.ModuleList(nn.BatchNorm1d(width) for _ in range(2))
        self.projections = nn.ModuleList(
            nn.Sequential(nn.Linear(width, width), nn.ELU(), nn.Linear(width, width))
            for _ in range(3)  # POI nodes, building nodes, fused regions
        )
        self.intra_score = nn.Parameter(torch.empty(width, width))
        self.inter_score = nn.Parameter(torch.empty(width, width))
        nn.init.xavier_uniform_(self.intra_score)
        nn.init.xavier_uniform_(self.inter_score)

    def forward(self, poi, building, walk, *, augmented=False):
        nodes = [encoder(graph.x, graph.edge_index, auger=augmented)
                 if graph.num_nodes else graph.x.new_zeros((0, self.width))
                 for encoder, graph in zip(self.encoders, (poi, building, walk), strict=True)]
        walk_regions = global_mean_pool(nodes[2], walk.batch, size=walk.num_graphs)
        views = (nodes[0][poi.dummy], nodes[1][building.dummy], walk_regions)
        available = torch.stack((poi.ptr.diff() > 1, building.ptr.diff() > 1, walk.ptr.diff() > 0), dim=-1)
        fused, weights = self.fusion(views, available)
        return self.norm[int(augmented)](fused), weights, views, nodes[:2]

    def loss(self, poi, building, walk, *, intra_weight=0.7):
        region, _, _, (p, b) = self(poi, building, walk)
        augmented, _, _, _ = self(poi, building, walk, augmented=True)
        p, b = self.projections[0](p), self.projections[1](b)
        block_ids = torch.arange(region.size(0), device=region.device)
        intra = local_to_region(p, poi, b, building, self.intra_score, block_ids)
        intra = intra + local_to_region(b, building, p, poi, self.intra_score, block_ids)
        # The walk encoder and all three attention scores receive gradients here.
        region_proj = self.projections[2]
        logits = region_proj(augmented) @ self.inter_score @ region_proj(region).T
        inter = contrastive_loss(logits, block_ids[:, None] == block_ids)
        return intra_weight * intra + (1 - intra_weight) * inter
