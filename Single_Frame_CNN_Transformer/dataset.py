from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import cv2
import numpy as np
from torchvision import transforms
import json


block_size = 16
downsampling_rate = 3

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

activities = ['Feeding', 'Preening', 'Swimming', 'Walking', 'Alert', 'Flying', 'Resting']

id_to_label = {i: activities[i] for i in range(len(activities))}
label_to_id = {v: k for k, v in id_to_label.items()}

distribution_path = 'annotations/splits.json'

def get_distribution():
    with open(distribution_path) as f:
        data = json.load(f)
    return data

class Dataset(Dataset):
    def __init__(self, image, labels):
        self.image = image
        self.labels = labels

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def __len__(self):
        return len(self.image)        
    
    def __getitem__(self, idx):

        image = self.image[idx].to(self.device)
        label = torch.tensor(self.labels[idx]).to(self.device)

        return image, label

def process_CSV(method, csv_path, data_distribution):
    
    df = pd.read_csv(csv_path, delimiter=';')
    dataset = {'train_set': {'frames': [], 'labels': []}, 'val_set': {'frames': [], 'labels': []}, 'test_set': {'frames': [], 'labels': []}}

    discarded = 0


    for index, row in df.iterrows():
        video_name = row['video_name']
        bird_id = row['bird_id']
        action_id = int(row['action_id'])
        start_frame = int(row['start_frame'])
        end_frame = int(row['end_frame'])

        action_frames = []

        # Downsample the dataset, to keep only one per 6 frames
        for i in range(start_frame, end_frame, downsampling_rate):
            action_frames.append(f'crops/{video_name}/{bird_id}/frame_{str(i).zfill(5)}.jpg')
        
        if len(action_frames) <= block_size:
            discarded += 1
            continue
        elif len(action_frames) >= 2 * block_size:
            # Divide the action in two or more blocks
            for i in range(0, len(action_frames), block_size):

                if i + block_size > len(action_frames):
                    break

                if method == 'central':
                    path = action_frames[i + block_size // 2]
                    embedding = cv2.imread(path)
                elif method == 'mean':
                    embedding = np.zeros((224, 224, 3), dtype=np.float32)

                    for j in range(i, i + block_size):
                        frame = cv2.imread(action_frames[j])
                        embedding += frame.astype(np.float32)
                    
                    embedding /= block_size
                    embedding = np.clip(embedding, 0, 255)
                    embedding = embedding.astype(np.uint8)
                
                embedding = cv2.cvtColor(embedding, cv2.COLOR_BGR2RGB)
                embedding = torch.tensor(embedding).permute(2, 0, 1).float() / 255.0
                embedding = normalize(embedding)
                

                if video_name in data_distribution['train_set']:
                    dataset['train_set']['frames'].append(embedding)
                    dataset['train_set']['labels'].append(action_id)
                elif video_name in data_distribution['val_set']:
                    dataset['val_set']['frames'].append(embedding)
                    dataset['val_set']['labels'].append(action_id)
                elif video_name in data_distribution['test_set']:
                    dataset['test_set']['frames'].append(embedding)
                    dataset['test_set']['labels'].append(action_id)


        elif len(action_frames) > block_size:

            action_frames = action_frames[:block_size]
            
            if method == 'central':
                path = action_frames[block_size // 2]        
                embedding = cv2.imread(path)
            elif method == 'mean':
                embedding = np.zeros((224, 224, 3), dtype=np.float32)
                
                for action_frames_path in action_frames:
                    frame = cv2.imread(action_frames_path)
                    embedding += frame.astype(np.float32)

                embedding /= block_size
                embedding = np.clip(embedding, 0, 255)
                embedding = embedding.astype(np.uint8)
            
            embedding = cv2.cvtColor(embedding, cv2.COLOR_BGR2RGB)
            embedding = torch.tensor(embedding).permute(2, 0, 1).float() / 255.0
            embedding = normalize(embedding)

            if video_name in data_distribution['train_set']:
                dataset['train_set']['frames'].append(embedding)
                dataset['train_set']['labels'].append(action_id)
            elif video_name in data_distribution['val_set']:
                dataset['val_set']['frames'].append(embedding)
                dataset['val_set']['labels'].append(action_id)
            elif video_name in data_distribution['test_set']:
                dataset['test_set']['frames'].append(embedding)
                dataset['test_set']['labels'].append(action_id)


    print('Included in dataset: train: ' + str(len(dataset['train_set']['frames'])) + ' val: ' + str(len(dataset['val_set']['frames'])) + ' test: ' + str(len(dataset['test_set']['frames'])))
    print('Samples discarted for dataset: ' + str(discarded))
    

    return dataset

def get_dataloader(method, batch_size=32, shuffle=True, csv_path = '/dataset/annotations/crops.csv'):

    data_distribution = get_distribution()

    data = process_CSV(method, csv_path, data_distribution)
    
    train_dataset = Dataset(data['train_set']['frames'], data['train_set']['labels'])
    val_dataset = Dataset(data['val_set']['frames'], data['val_set']['labels'])
    test_dataset = Dataset(data['test_set']['frames'], data['test_set']['labels'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader