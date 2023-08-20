from torch_geometric.utils import add_self_loops
import numpy as np
import os
import sys
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from src.utils import AslUtils
from src.dataset import AslDataset
from Model.model import BaseModel
from src.train import TrainModel
import mediapipe as mp
import cv2
import pandas as pd

mp_hands = mp.solutions.hands

path = os.getcwd()
sys.path.append(path)
csv_path = "Data\CSV"
save_path = "Data\raw"
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

utils = AslUtils()
# utils.create_train_csv(csv_path)
# utils.concat_all_csv(csv_path, save_path)


root_pt_path = os.path.join(path, "Data/")

train_ds = AslDataset(root_pt_path, "train.csv", test=False)
val_ds = AslDataset(root_pt_path, "test.csv", test=True)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl = DataLoader(val_ds, batch_size=1, shuffle=True)

model = BaseModel(3, 64, 32, 28).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
trainmodel = TrainModel(model, criterion, optimizer)

num_epochs = 100


train_losses = []
train_acc = []
test_losses = []
test_acc = []
for epoch in range(num_epochs):

    loss, acc = trainmodel.train_one_epoch(train_dl)
    train_losses.append(loss)
    train_acc.append(acc)

    print(f'Epoch : {epoch} | Train_Loss :{loss} | Accuracy :{acc}')

    loss, acc = trainmodel.test_one_epoch(test_dl)
    test_losses.append(loss)
    test_acc.append(acc)

    print(f'Epoch : {epoch} | Test_Loss :{loss} | Accuracy :{acc}')
