import mediapipe as mp
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torch_geometric.utils import add_self_loops


class AslDataset(Dataset):

    def __init__(self, root, file_name, test=False, transform=None, pre_transform=None):
        self.test = test
        self.file_name = file_name
        mp_hands = mp.solutions.hands
        hand_concc = mp_hands.HAND_CONNECTIONS
        source = []
        target = []
        for i, j in list(hand_concc):
            source.append(i)
            target.append(j)

        edge_index = add_self_loops(torch.tensor(
            np.array([source, target]), dtype=torch.int64))
        self.edge_index = edge_index[0]

        super(AslDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.file_name

    def len(self):
        return self.data.shape[0]

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index(drop=True)
        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]

        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def get(self, idx: int):
        if self.test:
            data = torch.load(os.path.join(
                self.processed_dir, f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(
                self.processed_dir, f'data_{idx}.pt'))
        return data

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index(drop=True)
        for idx, hand_pos in tqdm(self.data.iterrows(), total=self.data.shape[0]):

            x = torch.tensor(
                np.array(hand_pos.iloc[1:-1], dtype=np.float32), dtype=torch.float32).reshape(21, 3)

            y = torch.tensor(np.array(hand_pos.iloc[-1], dtype=np.int64))

            data = Data(x=x,
                        edge_index=self.edge_index,
                        y=y
                        )
            if self.test:
                torch.save(data, os.path.join(
                    self.processed_dir, f'data_test_{idx}.pt'))
            else:
                torch.save(data, os.path.join(
                    self.processed_dir, f'data_{idx}.pt'))
