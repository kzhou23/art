import torch
import numpy as np
import torch.utils.data as data


class ShapeNet_PC(data.Dataset):
    def __init__(self, path, num_points=2048, mode=0, splits=(0.85, 0.05, 0.1)):
        data = np.load(path)
        num_samples_train = int(data.shape[0] * splits[0])
        num_samples_vald = int(data.shape[0] * (splits[0]+splits[1]))
        self.num_points = num_points
        if mode == 0:
            self.data = data[:num_samples_train]
        elif mode == 1:
            self.data = data[num_samples_train: num_samples_vald]
        elif mode == 2:
            self.data = data[num_samples_vald:]
        self.data = torch.from_numpy(self.data).float()
        self.resample()

    def resample(self):
        if self.num_points < self.data.shape[1]:
            ind = torch.from_numpy(np.random.choice(self.data.shape[1], self.num_points)).long()
            self.pc = self.data[:, ind]
        else:
            self.pc = self.data

    def __getitem__(self, index):
        return self.pc[index]

    def __len__(self):
        return self.data.size(0)
