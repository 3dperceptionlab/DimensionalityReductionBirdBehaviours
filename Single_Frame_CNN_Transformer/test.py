import torch
import pytorch_lightning as pl
from torchvision.models import resnet50, mobilenet_v3_large, densenet201, vgg19, vit_b_32, swin_b
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import torch.nn as nn

from model import VideoClassificationModel
from dataset import get_dataloader, id_to_label

torch.set_float32_matmul_precision('medium')
pl.seed_everything(42, workers=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from sklearn.metrics import confusion_matrix

checkpoint = '/dataset/checkpoints/single_frame_central.ckpt'
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
    
    learning_rate = 0.00011396721935297006
    batch_size = 64
    model_name = 'densenet'
    method = 'central'
    class_weight = False

    _, _, test_dataloader = get_dataloader(method, batch_size=batch_size, shuffle=True)
    class_weights = torch.tensor([0.0849, 0.5297, 0.1976, 0.3531, 0.2488, 5.1205, 0.4655]).to(device)
    print(f"Class weights: {class_weights}")

    # Load model
    if model_name == 'resnet':
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, len(id_to_label))
    elif model_name == 'mobilenet':
        model = mobilenet_v3_large(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(id_to_label))
    elif model_name == 'densenet':
        model = densenet201(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, len(id_to_label))
    elif model_name == 'vgg':
        model = vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(id_to_label))
    elif model_name == 'vit':
        model = vit_b_32(pretrained=True)
        model.heads.head = nn.Linear(model.heads.head.in_features, len(id_to_label))
    elif model_name == 'swin':
        model = swin_b(pretrained=True)
        model.head = nn.Linear(model.head.in_features, len(id_to_label))
    else:
        print("Invalid model")
        return
    
    model = model.to(device)


    lightning_model = VideoClassificationModel.load_from_checkpoint(checkpoint_path=checkpoint, model=model, learning_rate=learning_rate, loss_weight=class_weights if class_weight else None)

    wandb_logger = WandbLogger(entity='CHAN-TWiN', project=project_name, log_model=True)

    early_stopping = EarlyStopping(monitor="val_accuracy_epoch", patience=15, mode="max")

    trainer = pl.Trainer(max_epochs=200, logger=wandb_logger, log_every_n_steps=5, callbacks=[early_stopping])

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