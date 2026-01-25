
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class ReadDataset(Dataset):

    def __init__(self,  source):
        self.data = torch.from_numpy(source).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def RandomSplit(datasets, train_set_percentage):
    train_len = int(len(datasets)*train_set_percentage)
    test_len = len(datasets)-train_len
    lengths = [train_len, test_len]
    return random_split(datasets, lengths)


def GetDataLoaders(npArray, batch_size, train_set_percentage = 0.9, 
                   shuffle=True, num_workers=4, pin_memory=True):
    
    pc = ReadDataset(npArray)

    if train_set_percentage>=1.0:
        train_loader = DataLoader(pc, shuffle=shuffle, num_workers=num_workers, 
                                  batch_size=batch_size, pin_memory=pin_memory, drop_last=True)
        test_loader = None
    else:
        train_set, test_set = RandomSplit(pc, train_set_percentage)
        train_loader = DataLoader(train_set, shuffle=shuffle, num_workers=num_workers, 
                                  batch_size=batch_size, pin_memory=pin_memory, drop_last=True)
        test_loader = DataLoader(test_set, shuffle=shuffle, num_workers=num_workers, 
                                 batch_size=batch_size, pin_memory=pin_memory, drop_last=False)
    
    return train_loader, test_loader

