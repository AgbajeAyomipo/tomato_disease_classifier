import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
plt.style.use("fivethirtyeight")
import PIL
from PIL import Image
from PIL import ImageFile
from matplotlib import image

import os, shutil, tqdm
from tqdm.auto import tqdm, trange
import pathlib
from pathlib import Path

import torch, torchvision, torchmetrics
import torch.nn as nn
from torchvision.transforms import v2 as v2
import lightning.pytorch as pl
from lightning.pytorch import LightningModule, LightningDataModule

ImageFile.LOAD_TRUNCATED_IMAGES = True
device = "cuda" if torch.cuda.is_available() else "cpu"


current_file = Path(__file__).resolve()
checkpoint_path = current_file.parent.parent / "checkpoints" / "epoch=14-step=12120.ckpt"

transform = v2.Compose(
        [
            v2.Resize(size = (224, 224)),
            v2.ToImage(),
            v2.ToDtype(dtype = torch.float32, scale = True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
idx_to_class = {0: 'Leaf mold',
                1: 'Tomato mosaic virus',
                2: 'Powdery mildew',
                3: 'Spider mites',
                4: 'Bacterial spot',
                5: 'Early blight',
                6: 'Healthy',
                7: 'Late blight',
                8: 'Tomato yellow leaf curl virus',
                9: 'Septoria leaf spot',
                10: 'Target spot'}

def prepare_image(img_path):
    image_ = transform(img_path).unsqueeze(0)
    return image_


def make_preds_return_class_class_confidence_dict(img, model):
    model.eval()
    with torch.inference_mode():
        logits = model(img)

    pred_probs = torch.softmax(logits, dim = 1)
    pred_probs_df = pd.DataFrame(data = torch.softmax(logits, dim = 1).numpy(), columns = idx_to_class.values())
    class_ = idx_to_class[torch.argmax(pred_probs, axis = 1).item()]
    pred_probs_df = pred_probs_df.T
    pred_probs_df.columns = ["confidence"]
    pred_probs_df = pred_probs_df.sort_values("confidence", ascending = False).head(5)
    label_dict = dict()
    for disease, confidence in zip(pred_probs_df.index, pred_probs_df["confidence"].values):
        label_dict[disease] = confidence
    return class_, label_dict


def load_model():
    class myLightningModel(pl.LightningModule):
        def __init__(self, model, lr):
            super().__init__()
            self.model = model
            self.lr = lr
            self.loss_fn = nn.CrossEntropyLoss()
            self.metric_fn = torchmetrics.classification.MulticlassAccuracy(num_classes = 11)
            self.save_hyperparameters(ignore = ["model"])

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            self.model.train()
            X, y = batch
            logits = self.model(X)
            loss = self.loss_fn(logits, y)
            acc = self.metric_fn(torch.flatten(torch.argmax(torch.softmax(logits, dim = 1), axis = 1)), y)
            self.log("Train accuracy", acc, prog_bar = True, on_epoch = True, on_step = False)
            self.log("Train logloss", loss, prog_bar = True, on_epoch = True, on_step = False)
            return {"Train Accuracy": acc, "loss": loss}

        def validation_step(self, batch, batch_idx):
            self.model.eval()
            X, y = batch
            logits = self.model(X)
            val_loss = self.loss_fn(logits, y)
            val_acc = self.metric_fn(torch.flatten(torch.argmax(torch.softmax(logits, dim = 1), axis = 1)), y)
            self.log("Val accuracy", val_acc, prog_bar = True, on_epoch = True, on_step = False)
            self.log("Val logloss", val_loss, prog_bar = True, on_epoch = True, on_step = False)
            return {"Val Accuracy": val_acc, "Val loss": val_loss}

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(params = self.model.parameters(), lr = self.lr, weight_decay = 1e-4)
            return optimizer
    
    model = torchvision.models.efficientnet.efficientnet_b0(progress = True, weights = torchvision.models.efficientnet.EfficientNet_B0_Weights.DEFAULT)
    model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=11, bias=True)
    )
    lightning_model = myLightningModel.load_from_checkpoint(checkpoint_path = checkpoint_path, map_location = device, model = model, lr = 1e-3)
    return lightning_model