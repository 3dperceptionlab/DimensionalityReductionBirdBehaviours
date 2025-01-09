import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import wandb
import sys

from model import VideoClassificationModel, DinoModel, HoGModel
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
    wandb.init(entity='CHAN-TWiN', project=project_name, config={"_disable_artifacts": True})
    config = wandb.config

    train_loader, val_dataloader, test_dataloader = get_dataloader(config.model, config.method, batch_size=config.batch_size, shuffle=True)
    class_weights = torch.tensor([0.0849, 0.5297, 0.1976, 0.3531, 0.2488, 5.1205, 0.4655]).to(device)
    print(f"Class weights: {class_weights}")

    if config.model == 'dino':
        model = DinoModel(num_classes=len(id_to_label), inner_dim=config.inner_dim, dropout=config.dropout, num_layers=config.num_layers)
    elif config.model == 'hog':
        model = HoGModel(num_classes=len(id_to_label), inner_dim=config.inner_dim, dropout=config.dropout, num_layers=config.num_layers)

    model = model.to(device)


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
        'num_layers': {'values': [2, 3, 4]},
        'method': {'values': ['mean', 'central']},
        'model': {'values': ['dino', 'hog']},
        'inner_dim': {'values': [256, 512, 1024]},
        'dropout': {'values': [0.1, 0.2, 0.3]},
        'class_weight': {'values': [True, False]},
    },
    'secondary_metrics': ['val_f1_epoch', 'val_precision_epoch', 'val_recall_epoch', 'test_f1_epoch', 'test_precision_epoch', 'test_recall_epoch', 'val_accuracy_epoch', 'test_accuracy_epoch', 'train_accuracy_epoch']
}

sweep_id = wandb.sweep(sweep_config, entity='CHAN-TWiN', project=project_name)
wandb.agent(sweep_id, entity='CHAN-TWiN', project=project_name, function=train)