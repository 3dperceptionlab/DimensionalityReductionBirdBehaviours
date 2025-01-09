import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import wandb

from model import VideoClassificationModel, MLPModel
from dataset import get_dataloader, id_to_label

torch.set_float32_matmul_precision('medium')
pl.seed_everything(42, workers=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    wandb.init(entity='CHAN-TWiN', project=project_name)
    config = wandb.config

    train_loader, val_dataloader, test_dataloader = get_dataloader(config.model, batch_size=config.batch_size, shuffle=True)
    class_weights = torch.tensor([0.0849, 0.5297, 0.1976, 0.3531, 0.2488, 5.1205, 0.4655]).to(device)
    print(f"Class weights: {class_weights}")

    # Load model
    model = MLPModel(input_size=400, num_classes=len(id_to_label), hidden_dim=config.hidden_dim, dropout=config.dropout, num_layers=config.num_layers)

    lightning_model = VideoClassificationModel(model=model, learning_rate=config.learning_rate, loss_weight=class_weights if config.class_weight else None)

    wandb_logger = WandbLogger(entity='CHAN-TWiN', project=project_name, log_model=True)

    early_stopping = EarlyStopping(monitor="val_accuracy_epoch", patience=15, mode="max")

    trainer = pl.Trainer(max_epochs=200, logger=wandb_logger, log_every_n_steps=5, callbacks=[early_stopping])
    trainer.fit(lightning_model, train_loader, val_dataloader)
    trainer.test(lightning_model, test_dataloader)

sweep_config = {
    'method': 'random',
    'metric': {'name': 'test_accuracy_epoch', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'distribution': 'uniform', 'min': 0.0001, 'max': 0.01},
        'batch_size': {'values': [16,32,64]},
        'hidden_dim': {'values': [256, 512, 1024]},
        'dropout': {'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
        'num_layers': {'values': [1, 2, 3, 4, 5]},
        'model': {'values': ['r3d', 'mvit', 's3d', 'swin']},
        'class_weight': {'values': [True, False]},
    },
    'secondary_metrics': ['val_f1_epoch', 'val_precision_epoch', 'val_recall_epoch', 'test_f1_epoch', 'test_precision_epoch', 'test_recall_epoch', 'val_accuracy_epoch', 'test_accuracy_epoch', 'train_accuracy_epoch']
}

sweep_id = wandb.sweep(sweep_config, entity='CHAN-TWiN', project=project_name)
wandb.agent(sweep_id, entity='CHAN-TWiN', project=project_name, function=train)