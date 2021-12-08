import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RegardBertDataset(Dataset):
    def __init__(self, encodings, labels=[]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if len(self.labels) > 0:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def get_dataloader(X, Y, batch_size, shuffle=True):
    if isinstance(Y, pd.Series):
        Y = Y.values.astype("int")

    data = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))

    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=mp.cpu_count(),
    )
    return dataloader

