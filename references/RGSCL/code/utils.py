import torch
from typing import List, Literal


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = .0):
        """
        Parameters:
        patience (int): Number of epochs to wait after last time validation loss improved.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def calc_eval_result_per_region(region_ids, labels, predictions, task: Literal['func', 'price', 'checkin']) -> List:
    predictions = torch.tensor(predictions).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    labels = torch.tensor(labels).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    if task == 'func':
        correct = (predictions == labels).int()
    elif task == 'price' or 'checkin':
        correct = torch.abs(predictions - labels)
    region_predictions_list = []
    for _region_id, _correct in zip(region_ids, correct):
        region_predictions_list.append([_region_id.item(), _correct.item()])
    return region_predictions_list
