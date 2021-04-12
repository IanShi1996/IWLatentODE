import torch
from torch.utils.data import Dataset, DataLoader

from utils import gpu

# Hard coded, since datasets are included with repo
DATA_PATH_DICT = {
    "sine": "./data/sine_data_2021-04-09 00:13:41.249505",
    "aussign": "./data/aussign_parsed",
}


class GenericSet(Dataset):
    def __init__(self, data, time):
        self.data = data
        self.time = time

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.time


class SineSet(Dataset):
    """ Maintained for compatibility. """
    def __init__(self, data, time):
        self.data = data
        self.time = time

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.time


def get_dataloaders(dataset_type, batch_size):
    data_path = DATA_PATH_DICT[dataset_type]

    if dataset_type == "sine":
        generator = torch.load(data_path)['generator']

        train_time, train_data = generator.get_train_set()
        val_time, val_data = generator.get_val_set()

        train_data = train_data.reshape(len(train_data), -1, 1)
        val_data = val_data.reshape(len(val_data), -1, 1)

        train_set = SineSet(gpu(train_data), gpu(train_time))
        val_set = SineSet(gpu(val_data), gpu(val_time))

    elif dataset_type == "aussign":
        data = torch.load(data_path)

        train_data = data["train_dataset"]
        val_data = data["val_dataset"]

        train_time = list(range(train_data.shape[1]))
        val_time = list(range(val_data.shape[1]))

        train_set = GenericSet(gpu(train_data), gpu(train_time))
        val_set = GenericSet(gpu(val_data), gpu(val_time))
    else:
        raise ValueError("Unknown dataset type.")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=len(val_set))

    return train_loader, val_loader
