import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch.nn as nn
import sys

from model import VideoClassificationModel, DinoModel, HoGModel
from dataset import get_dataloader, id_to_label

torch.set_float32_matmul_precision('medium')
pl.seed_everything(42, workers=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

project_name = 'dimensionality_reduction'

from sklearn.metrics import confusion_matrix
checkpoint = '/dataset/checkpoints/hog.ckpt'

def calculate_class_weights(dataloader, num_classes):
    class_counts = torch.zeros(num_classes).to(device)
    for _, label in dataloader:
        label = label.view(-1).to(device)
        class_counts[label] += 1
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    return class_weights

def train():
    learning_rate = 0.0018506656240730576
    batch_size = 16
    num_layers = 4
    method = 'mean'
    model_name = 'hog'
    inner_dim = 1024
    dropout = 0.1
    class_weight = False

    _, _, test_dataloader = get_dataloader(model_name, method, batch_size=batch_size, shuffle=True)
    class_weights = torch.tensor([0.0849, 0.5297, 0.1976, 0.3531, 0.2488, 5.1205, 0.4655]).to(device)
    print(f"Class weights: {class_weights}")

    if model_name == 'dino':
        model = DinoModel(num_classes=len(id_to_label), inner_dim=inner_dim, dropout=dropout, num_layers=num_layers)
    elif model_name == 'hog':
        model = HoGModel(num_classes=len(id_to_label), inner_dim=inner_dim, dropout=dropout, num_layers=num_layers)

    model = model.to(device)

    lightning_model = VideoClassificationModel.load_from_checkpoint(checkpoint_path=checkpoint, model=model, learning_rate=learning_rate, loss_weight=class_weights if class_weight else None)

    wandb_logger = WandbLogger(entity='CHAN-TWiN', project=project_name, log_model=True)
    trainer = pl.Trainer(max_epochs=200, logger=wandb_logger, log_every_n_steps=5)
    
    lightning_model.eval()  # Set the model to evaluation mode
    lightning_model.model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculation for testing
        for batch in test_dataloader:
            inputs, labels = batch

            outputs = lightning_model(inputs)
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Generate the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    
    trainer.test(lightning_model, test_dataloader)

train()
