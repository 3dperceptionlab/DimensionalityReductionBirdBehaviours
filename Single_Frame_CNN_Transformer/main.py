import torch
import pytorch_lightning as pl
from torchvision.models import resnet50, mobilenet_v3_large, densenet201, vgg19, vit_b_32, swin_b
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import wandb
import torch.nn as nn
import sys

from model import VideoClassificationModel
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

    train_loader, val_dataloader, test_dataloader = get_dataloader(config.method, batch_size=config.batch_size, shuffle=True)
    class_weights = torch.tensor([0.0849, 0.5297, 0.1976, 0.3531, 0.2488, 5.1205, 0.4655]).to(device)
    print(f"Class weights: {class_weights}")

    # Load model
    if config.model == 'resnet':
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, len(id_to_label))
    elif config.model == 'mobilenet':
        model = mobilenet_v3_large(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(id_to_label))
    elif config.model == 'densenet':
        model = densenet201(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, len(id_to_label))
    elif config.model == 'vgg':
        model = vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(id_to_label))
    elif config.model == 'vit':
        model = vit_b_32(pretrained=True)
        model.heads.head = nn.Linear(model.heads.head.in_features, len(id_to_label))
    elif config.model == 'swin':
        model = swin_b(pretrained=True)
        model.head = nn.Linear(model.head.in_features, len(id_to_label))
    else:
        print("Invalid model")
        return
    
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
        'model': {'values': ['resnet', 'mobilenet', 'densenet', 'vgg', 'vit', 'swin']},
        'method': {'values': ['mean', 'central']},
        'class_weight': {'values': [True, False]},
    },
    'secondary_metrics': ['val_f1_epoch', 'val_precision_epoch', 'val_recall_epoch', 'test_f1_epoch', 'test_precision_epoch', 'test_recall_epoch', 'val_accuracy_epoch', 'test_accuracy_epoch', 'train_accuracy_epoch']
}

sweep_id = wandb.sweep(sweep_config, entity='CHAN-TWiN', project=project_name)
wandb.agent(sweep_id, entity='CHAN-TWiN', project=project_name, function=train)