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
import gradio as gr

import torch, torchvision
import torch.nn as nn
from torchvision.transforms import v2 as v2
import lightning.pytorch as pl
from lightning.pytorch import LightningModule, LightningDataModule

ImageFile.LOAD_TRUNCATED_IMAGES = True

import utils
from utils.utils import prepare_image, make_preds_return_class_class_confidence_dict, load_model

def run_gradio_app():
    def return_class_x_label_conf_dict(img_path_):
        lightning_model = load_model()
        image = prepare_image(img_path = img_path_)
        class_, label_conf_dict = make_preds_return_class_class_confidence_dict(img = image, model = lightning_model)
        return f"Predicted class: {class_}", label_conf_dict
    
    title = "Tomato Leaf Disease Classification with Prediction Confidence Visualization"
    description = "This intuitive interface simplifies the process of diagnosing diseases in tomato plants using leaf imagery. \
                   By uploading a clear image of a tomato leaf, the application leverages a trained deep learning classifier to \
                   identify the most likely disease affecting the plant. The system returns the top predicted class, along with a \
                   ranked list of the top 5 possible diseases and their associated confidence scores. This layered feedback ensures \
                   users not only receive a diagnosis but also understand the certainty of each prediction.\n \
                    - Please ensure that uploaded images are of individual leaves with minimal background clutter for optimal accuracy.\n \
                    - The model was trained on a curated dataset of common tomato leaf diseases and performs best on clear, close-up images."

    demo = gr.Interface(fn = return_class_x_label_conf_dict, inputs = [gr.Image(type = "pil", label = "Upload Image of Tomato Leaf here:")],
                        outputs=[gr.Textbox(label = "Predicted Disease Class"), gr.Label(label = "Top 5 Predicted Class Probability distribution")],
                        title = title,
                        description=description,
                        theme = gr.themes.Ocean())
    demo.launch(share = False)


if __name__ == "__main__":
    run_gradio_app()