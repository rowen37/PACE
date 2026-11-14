"""Relation-aware block encoder with interchangeable attention operators."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv

from data import RELATIONS

WIDTH = 16
EMBEDDING_DIM = 6
MODEL_SEEDS = (13, 29, 53)


class BlockEncoder(nn.Module):
    def __init__(self, num_categories, operator, width=WIDTH, dropout=0, block_features=3):
        super().__init__()
        self.inputs = nn.ModuleDict({key: nn.Linear(dim, width)
                                    for key, dim in {"place": num_categories, "building": 3,
                                                     "walk": 3, "block": block_features}.items()})
        self.layers = nn.ModuleList([
            HeteroConv({rel: operator((width, width), width, heads=2, concat=False,
                                       add_self_loops=False, bias=False) for rel in RELATIONS},
                       aggr="sum")
            for _ in range(2)
        ])
        self.embedding = nn.Linear(width, EMBEDDING_DIM)
        self.head = nn.Linear(EMBEDDING_DIM, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        x = {key: linear(batch[key].x) for key, linear in self.inputs.items()}
        for layer in self.layers:
            updates = layer(x, batch.edge_index_dict)
            x = {key: self.dropout(F.elu(value + updates[key])) for key, value in x.items()}
        embedding = self.embedding(x["block"])
        return self.head(embedding).squeeze(-1)


def graph_checkpoints(train, test, num_categories, operator, checkpoints, seed,
                      *, width=WIDTH, lr=0.004, weight_decay=0.01, dropout=0,
                      loss_name="bce", track_training=False, auxiliary_loss=None):
    torch.manual_seed(seed)
    model = BlockEncoder(num_categories, operator, width, dropout, train[0]["block"].x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    objectives = {"bce": F.binary_cross_entropy_with_logits,
                  "mae": lambda logits, target: F.l1_loss(logits.sigmoid(), target)}
    objective = objectives[loss_name]
    train_batches = DataLoader(train, batch_size=24, shuffle=True,
                               generator=torch.Generator().manual_seed(seed))
    test_batches = list(DataLoader(test, batch_size=24))
    train_eval_batches = list(DataLoader(train, batch_size=24)) if track_training else []
    predictions = {}
    train_predictions = {}
    for epoch in range(1, max(checkpoints) + 1):
        model.train()
        for batch in train_batches:
            optimizer.zero_grad()
            loss = objective(model(batch), batch.y)
            if auxiliary_loss is not None:
                loss = loss + auxiliary_loss(model, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
        if epoch in checkpoints:
            model.eval()
            with torch.no_grad():
                predictions[epoch] = torch.cat([model(batch).sigmoid() for batch in test_batches]).numpy()
                if track_training:
                    train_predictions[epoch] = torch.cat(
                        [model(batch).sigmoid() for batch in train_eval_batches]).numpy()
    return predictions, sum(p.numel() for p in model.parameters()), train_predictions


def fit_graph(train, test, num_categories, operator, epochs, seed):
    predictions, params, _ = graph_checkpoints(train, test, num_categories, operator, (epochs,), seed)
    return predictions[epochs], params
