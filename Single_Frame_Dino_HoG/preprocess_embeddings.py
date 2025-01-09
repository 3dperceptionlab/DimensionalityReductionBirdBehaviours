import pandas as pd
import torch
import cv2
import numpy as np
from torchvision import transforms
from transformers import Dinov2Model

block_size = 16
downsampling_rate = 3

activities = ['Feeding', 'Preening', 'Swimming', 'Walking', 'Alert', 'Flying', 'Resting']

id_to_label = {i: activities[i] for i in range(len(activities))}
label_to_id = {v: k for k, v in id_to_label.items()}

distribution_path = 'annotations/splits.json'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

csv_path = '/dataset/annotations/crops.csv'

    
df = pd.read_csv(csv_path, delimiter=';')
actions = np.array([])

model_mames = ['hog', 'dino']
method_name = ['central', 'mean']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device)
model.eval()

hog_descriptor = cv2.HOGDescriptor()


for model_name in model_mames:

    for method in method_name:

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

                    if model_name == 'hog':
                        hog_image = (embedding.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                        hog_features = hog_descriptor.compute(hog_image)
                        tensor = torch.tensor(hog_features).float().cpu()

                    elif model_name == 'dino':
                        frames = embedding.unsqueeze(0).to(device)
                        tensor = model(frames).last_hidden_state[0][0].cpu()

                    path = f'embeddings/{model_name}/{method}/{video_name}_{bird_id}_{start_frame}_{i}.pt'
                    torch.save(tensor, path)

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

                if model_name == 'hog':
                    hog_image = (embedding.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                    hog_features = hog_descriptor.compute(hog_image)
                    tensor = torch.tensor(hog_features).float().cpu()

                elif model_name == 'dino':
                    frames = embedding.unsqueeze(0).to(device)
                    tensor = model(frames).last_hidden_state[0][0].cpu()

                path = f'embeddings/{model_name}/{method}/{video_name}_{bird_id}_{start_frame}.pt'
                torch.save(tensor, path)