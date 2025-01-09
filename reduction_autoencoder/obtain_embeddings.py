from train_auto_encoder_reduction import VideoAutoencoder

import torch
import pandas as pd
import cv2
import numpy as np
from torchvision import transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

path = '/dataset/autoencoder/trained_autoencoder.pth'

autoencoder = VideoAutoencoder()
autoencoder.load_state_dict(torch.load(path))
# Keep only the encoder
encoder = autoencoder.encoder
encoder = encoder.to(device)
encoder.eval()

csv_path = '/dataset/annotations/crops.csv'
df = pd.read_csv(csv_path, delimiter=';')

block_size = 16
downsampling_rate = 3

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

            frame_paths = action_frames[i:i+block_size]
            frames = []
            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            frames = np.array(frames)
            frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)

            frames = frames.float() / 255.0  # Normalize pixel values to [0, 1]
            frames = normalize(frames)

            frames = frames.permute(1,0,2,3)
            frames = frames.unsqueeze(0)

            frames = frames.to(device)
            
            with torch.no_grad():
                latent = encoder(frames)
                latent = latent.squeeze().cpu()
                path = f'/dataset/autoencoder/embeddings/{video_name}_{bird_id}_{start_frame}_{i}.pt'
                torch.save(latent, path)


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
        frames = frames.unsqueeze(0)

        frames = frames.to(device)

        with torch.no_grad():
            latent = encoder(frames)
            latent = latent.squeeze().cpu()
            path = f'/dataset/autoencoder/embeddings/{video_name}_{bird_id}_{start_frame}.pt'
            torch.save(latent, path)
