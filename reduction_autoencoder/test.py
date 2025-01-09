import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import sys

from autoencoder_model import VideoClassificationModel, EmbeddingModel
from autoencoder_dataset import get_dataloader, id_to_label

torch.set_float32_matmul_precision('medium')
pl.seed_everything(42, workers=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
project_name = 'dimensionality_reduction'

from sklearn.metrics import confusion_matrix

checkpoint = '/dataset/checkpoints/autoencoder.ckpt'

def calculate_class_weights(dataloader, num_classes):
    class_counts = torch.zeros(num_classes).to(device)
    for _, label in dataloader:
        label = label.view(-1).to(device)
        class_counts[label] += 1
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    return class_weights

def train():

    config = {
        'learning_rate': 0.006663075069938562,
        'batch_size': 64,
        'inner_dim': 256,
        'dropout': 0.3,
        'num_layers': 3,
        'model': 'autoencoder',
        'class_weight': False
    }


    _, _, test_dataloader = get_dataloader(batch_size=config['batch_size'], shuffle=True)
    class_weights = torch.tensor([0.0849, 0.5297, 0.1976, 0.3531, 0.2488, 5.1205, 0.4655]).to(device)
    print(f"Class weights: {class_weights}")

    if config['model'] == 'autoencoder':
        model = EmbeddingModel(num_classes=len(id_to_label), inner_dim=config['inner_dim'], dropout=config['dropout'], num_layers=config['num_layers'])
    else:
        raise ValueError(f"Model {config['model']} not supported")

    lightning_model = VideoClassificationModel.load_from_checkpoint(checkpoint_path=checkpoint, model=model, learning_rate=config['learning_rate'], loss_weight=class_weights if config['class_weight'] else None)

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