import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append('modules')

from dataset import get_dataloader

# Encoder
class VideoEncoder(nn.Module):
    def __init__(self):
        super(VideoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),  # Temporal stride = 1
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),  # Temporal stride = 2
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),  # Temporal stride = 2
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),  # Temporal stride = 2
            nn.BatchNorm3d(512),
            nn.ReLU(),
        )
        self.fc = nn.Linear(512 * 2 * 14 * 14, 1024)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

class VideoDecoder(nn.Module):
    def __init__(self):
        super(VideoDecoder, self).__init__()
        self.fc = nn.Linear(1024, 512 * 2 * 14 * 14)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1, output_padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.ConvTranspose3d(256, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1, output_padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1, output_padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 3, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),  # Last layer
            nn.Sigmoid(),  # Normalized output
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 2, 14, 14)  # Adjust to match decoder input
        x = self.decoder(x)
        return x

# Full Autoencoder
class VideoAutoencoder(nn.Module):
    def __init__(self):
        super(VideoAutoencoder, self).__init__()
        self.encoder = VideoEncoder()
        self.decoder = VideoDecoder()

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
def train_spatiotemporal_autoencoder(dataloader, epochs, lr, device):
    autoencoder = VideoAutoencoder().to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        autoencoder.train()
        running_loss = 0.0
        for batch_idx, (inputs, _) in enumerate(dataloader):  # Replace `dataloader` with your DataLoader
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}", flush=True)

        # Each 20 epochs, save the model
        if (epoch + 1) % 20 == 0:
            torch.save(autoencoder.state_dict(), f'autoencoder/trained_autoencoder_{epoch + 1}.pth')
    
    return autoencoder

def get_latent_representations(autoencoder, dataloader, device='cuda'):
    autoencoder.eval()
    latent_representations = []
    with torch.no_grad():
        for frames, _ in dataloader:
            frames = frames.to(device)
            latent, _ = autoencoder(frames)
            latent_representations.append(latent.cpu().numpy())
    return np.vstack(latent_representations)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, _, _ = get_dataloader(batch_size=32, shuffle=True)
    autoencoder = train_spatiotemporal_autoencoder(train_loader, epochs=100, lr=0.0001, device=device)

    # Save the trained autoencoder
    torch.save(autoencoder.state_dict(), 'autoencoder/trained_autoencoder.pth')

if __name__ == '__main__':
    main()