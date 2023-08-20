from torch_geometric.utils import add_self_loops
import numpy as np
import os
import torch
from src.utils import AslUtils
from Model.model import BaseModel
import mediapipe as mp
import cv2
import pandas as pd
from torch_geometric.data import Data
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
path = os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
utils = AslUtils()


def _getlabel(num):
    labels = os.listdir('asl_dataset')
    label_dict = dict(zip(np.arange(len(labels)), labels))
    return label_dict[num]


def predict(img):
    img = np.array(img)
    model = BaseModel(3, 64, 32, 28).to(device)
    model_path = os.path.join(path, "Model/base_model.pt")
    torch.save(model.state_dict(), model_path)
    hand_concc = mp_hands.HAND_CONNECTIONS
    source = []
    target = []

    for i, j in list(hand_concc):
        source.append(i)
        target.append(j)

    edge_index = add_self_loops(torch.tensor(
        np.array([source, target]), dtype=torch.int64))
    edge_index = edge_index[0]

    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.4
    )
    img_result = hands.process(img)  # the graph connections
    img_list = []
    temp_img = img.copy()
    if img_result.multi_hand_landmarks is not None:
        for landmarks in img_result.multi_hand_landmarks:
            img_list.append(utils.get_hand_landmark(
                "img.jpg", landmarks, 0))
            mp_draw.draw_landmarks(temp_img, landmarks,
                                   mp_hands.HAND_CONNECTIONS)
    df = pd.DataFrame.from_dict(img_list)
    x = torch.tensor(df.loc[0][1:-1], dtype=torch.float32).reshape(21, 3)
    out = model(x, edge_index, torch.tensor([0]))
    return temp_img, _getlabel(out.argmax(dim=1).item())
