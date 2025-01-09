from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import numpy as np
from torchvision import transforms
import json


block_size = 16
downsampling_rate = 3

activities = ['Feeding', 'Preening', 'Swimming', 'Walking', 'Alert', 'Flying', 'Resting']

id_to_label = {i: activities[i] for i in range(len(activities))}
label_to_id = {v: k for k, v in id_to_label.items()}

distribution_path = 'annotations/splits.json'

def get_distribution():
    with open(distribution_path) as f:
        data = json.load(f)
    return data

class Dataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def __len__(self):
        return len(self.embeddings)        
    
    def __getitem__(self, idx):

        embedding = self.embeddings[idx].to(self.device)
        label = torch.tensor(self.labels[idx]).to(self.device)

        return embedding, label

def process_CSV(model, csv_path, data_distribution):
    
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

                path = f'/dataset/MDPI-ReduccionDimensionalidad/embeddings/{model}/{video_name}_{bird_id}_{start_frame}_{i}.pt'
                embedding = torch.load(path)

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
            
            path = f'/dataset/MDPI-ReduccionDimensionalidad/embeddings/{model}/{video_name}_{bird_id}_{start_frame}.pt'
            embedding = torch.load(path)

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

def get_dataloader(model, batch_size=32, shuffle=True, csv_path = '/dataset/annotations/crops.csv'):

    data_distribution = get_distribution()

    data = process_CSV(model, csv_path, data_distribution)
    
    train_dataset = Dataset(data['train_set']['frames'], data['train_set']['labels'])
    val_dataset = Dataset(data['val_set']['frames'], data['val_set']['labels'])
    test_dataset = Dataset(data['test_set']['frames'], data['test_set']['labels'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader