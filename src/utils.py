import mediapipe as mp
from pathlib import Path
import os
import cv2
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


BASEDIR = Path(__file__).resolve(strict=True).parent.parent
sys.path.append(BASEDIR)


class AslUtils(object):
    def __init__(self):
        self.basedir = BASEDIR
        self.train_image_dir = "asl_dataset"

    def get_hand_landmark(self, img_name, landmarks, label):
        coords = {
            "Image_Name": img_name,

            "0: WRIST_x": landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
            "0: WRIST_y": landmarks.landmark[mp_hands.HandLandmark.WRIST].y,
            "0: WRIST_z": landmarks.landmark[mp_hands.HandLandmark.WRIST].z,

            "1: THUMB_CMC_x": landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x,
            "1: THUMB_CMC_y": landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y,
            "1: THUMB_CMC_z": landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z,

            "2: THUMB_MCP_x": landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x,
            "2: THUMB_MCP_y": landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y,
            "2: THUMB_MCP_z": landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z,

            "3: THUMB_IP_x": landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x,
            "3: THUMB_IP_y": landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y,
            "3: THUMB_IP_z": landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z,

            "4: THUMB_TIP_x": landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
            "4: THUMB_TIP_y": landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
            "4: THUMB_TIP_z": landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z,

            "5: INDEX_FINGER_MCP_x": landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
            "5: INDEX_FINGER_MCP_y": landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
            "5: INDEX_FINGER_MCP_z": landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z,

            "6: INDEX_FINGER_PIP_x": landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
            "6: INDEX_FINGER_PIP_y": landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
            "6: INDEX_FINGER_PIP_z": landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z,

            "7: INDEX_FINGER_DIP_x": landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x,
            "7: INDEX_FINGER_DIP_y": landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y,
            "7: INDEX_FINGER_DIP_z": landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z,

            "8: INDEX_FINGER_TIP_x": landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
            "8: INDEX_FINGER_TIP_y": landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
            "8: INDEX_FINGER_TIP_z": landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z,

            "9: MIDDLE_FINGER_MCP_x": landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
            "9: MIDDLE_FINGER_MCP_y": landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
            "9: MIDDLE_FINGER_MCP_z": landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z,

            "10: MIDDLE_FINGER_PIP_x": landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x,
            "10: MIDDLE_FINGER_PIP_y": landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
            "10: MIDDLE_FINGER_PIP_z": landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z,

            "11: MIDDLE_FINGER_DIP_x": landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x,
            "11: MIDDLE_FINGER_DIP_y": landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y,
            "11: MIDDLE_FINGER_DIP_z": landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z,

            "12: MIDDLE_FINGER_TIP_x": landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
            "12: MIDDLE_FINGER_TIP_y": landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
            "12: MIDDLE_FINGER_TIP_z": landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z,

            "13: RING_FINGER_MCP_x": landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x,
            "13: RING_FINGER_MCP_y": landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y,
            "13: RING_FINGER_MCP_z": landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z,

            "14: RING_FINGER_PIP_x": landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x,
            "14: RING_FINGER_PIP_y": landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
            "14: RING_FINGER_PIP_z": landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z,

            "15: RING_FINGER_DIP_x": landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x,
            "15: RING_FINGER_DIP_y": landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y,
            "15: RING_FINGER_DIP_z": landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z,

            "16: RING_FINGER_TIP_x": landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x,
            "16: RING_FINGER_TIP_y": landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y,
            "16: RING_FINGER_TIP_z": landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z,

            "17: PINKY_MCP_x": landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x,
            "17: PINKY_MCP_y": landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y,
            "17: PINKY_MCP_z": landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z,

            "18: PINKY_PIP_x": landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x,
            "18: PINKY_PIP_y": landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y,
            "18: PINKY_PIP_z": landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z,

            "19: PINKY_DIP_x": landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x,
            "19: PINKY_DIP_y": landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y,
            "19: PINKY_DIP_z": landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z,

            "20: PINKY_TIP_x": landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x,
            "20: PINKY_TIP_y": landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y,
            "20: PINKY_TIP_z": landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z,

            "Label": label
        }
        return coords

    def read_img_using_mediapipe(self, img_dir):
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        results = hands.process(img)
        return results

    def create_train_csv(self, csv_folder):
        train_path = os.path.join(self.basedir, self.train_image_dir)
        save_csv = os.path.join(self.basedir, csv_folder)
        labels = os.listdir(train_path)
        label_dict = dict(zip(labels, np.arange(len(labels))))

        for index, label in enumerate(labels):
            class_label = label_dict[label]
            label_path = os.path.join(train_path, label)
            img_list = []
            for img in tqdm(os.listdir(label_path), total=len(os.listdir(label_path))):
                img_name = os.path.join(label_path, img)
                img_result = self.read_img_using_mediapipe(img_name)

                if img_result.multi_hand_landmarks is not None:
                    for landmarks in img_result.multi_hand_landmarks:
                        img_list.append(self.get_hand_landmark(
                            img_name, landmarks, class_label))

            df = pd.DataFrame.from_dict(img_list)
            df = df.reset_index(drop=True)
            df.to_csv(os.path.join(save_csv, f"{label}.csv"), index=False)

    def concat_all_csv(self, csv_path, save_path, shuffle=True, test_size=0.15):
        all_csv_path = os.path.join(self.basedir, csv_path)
        all_save_path = os.path.join(self.basedir, "Data/raw")
        # print(all_csv_path)
        # print(all_save_path)
        concatenated_csv_file = pd.concat(pd.read_csv(os.path.join(
            all_csv_path, file)) for file in os.listdir(all_csv_path))

        if "Unnamed: 0" in concatenated_csv_file.columns:
            concatenated_csv_file.drop(['Unnamed: 0'], axis=1, inplace=True)

        if shuffle:
            concatenated_csv_file = concatenated_csv_file.sample(frac=1)

        split_index = int(test_size * len(concatenated_csv_file))

        concatenated_csv_file.to_csv(os.path.join(
            all_save_path, 'concatenated.csv'), index=False)

        concatenated_csv_file[split_index:].to_csv(os.path.join(
            all_save_path, 'train.csv'), index=False)

        concatenated_csv_file[:split_index].to_csv(os.path.join(
            all_save_path, 'test.csv'), index=False)
