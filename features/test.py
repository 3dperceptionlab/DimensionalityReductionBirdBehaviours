import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from model import VideoClassificationModel, MLPModel
from dataset import get_dataloader, id_to_label
import torch.nn as nn
from sklearn.metrics import confusion_matrix

torch.set_float32_matmul_precision('medium')
pl.seed_everything(42, workers=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = '/dataset/checkpoints/features.ckpt'
project_name = 'dimensionality_reduction'


def calculate_class_weights(dataloader, num_classes):
    class_counts = torch.zeros(num_classes).to(device)
    for _, label in dataloader:
        label = label.view(-1).to(device)
        class_counts[label] += 1
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    return class_weights

def train():

    learning_rate = 0.0046722
    batch_size = 32
    hidden_dim = 256
    dropout = 0.4
    num_layers = 5
    model_mame = 'swin'
    class_weight = False

    _, _, test_dataloader = get_dataloader(model_mame, batch_size=batch_size, shuffle=True)
    class_weights = torch.tensor([0.0849, 0.5297, 0.1976, 0.3531, 0.2488, 5.1205, 0.4655]).to(device)
    print(f"Class weights: {class_weights}")

    # Load model
    model = MLPModel(input_size=400, num_classes=len(id_to_label), hidden_dim=hidden_dim, dropout=dropout, num_layers=num_layers)
    model = model.to(device)

    # Load from checkpoint
    lightning_model = VideoClassificationModel.load_from_checkpoint(checkpoint, model=model, learning_rate=learning_rate, loss_weight=class_weights if class_weight else None)
    lightning_model = lightning_model.to(device)

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

