import pandas as pd
import torch
import cv2
import numpy as np
from torchvision import transforms

from torchvision.models.video import r3d_18, mvit_v2_s, s3d, swin3d_s

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

model_mames = ['r3d', 'mvit', 's3d', 'swin']

for model_name in model_mames:

    if model_name == 'r3d':
        model = r3d_18(pretrained=True)
    elif model_name == 'mvit':
        model = mvit_v2_s(pretrained=True)
    elif model_name == 's3d':
        model = s3d(pretrained=True)
    elif model_name == 'swin':
        model = swin3d_s(pretrained=True)
    else:
        raise ValueError(f'Invalid model name: {model_name}')
    
    model = model.to(device)
    model.eval()

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

                frames = []
                for frame_path in action_frames[i:i+block_size]:
                    frame = cv2.imread(frame_path)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

                frames = np.array(frames)
                frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)

                frames = frames.float() / 255.0  # Normalize pixel values to [0, 1]
                frames = normalize(frames)

                frames = frames.permute(1,0,2,3)
                frames = frames.unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(frames)
                    output = output.squeeze(0)
                    output = output.cpu()

                    path = f'embeddings/{model_name}/{video_name}_{bird_id}_{start_frame}_{i}.pt'
                    torch.save(output, path)

        elif len(action_frames) > block_size:
            action_frames = action_frames[:block_size]

            frames = []
            for frame_path in action_frames:
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            frames = np.array(frames)
            frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)

            frames = frames.float() / 255.0  # Normalize pixel values to [0, 1]
            frames = normalize(frames)

            frames = frames.permute(1,0,2,3)
            frames = frames.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(frames)
                output = output.squeeze(0)
                output = output.cpu()

                path = f'embeddings/{model_name}/{video_name}_{bird_id}_{start_frame}.pt'
                torch.save(output, path)