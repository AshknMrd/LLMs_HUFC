from pathlib import Path

import torch
import gradio as gr
from torch import nn
from PIL import Image
import numpy as np

LABELS = Path('class_names.txt').read_text().splitlines()

model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(1152, 256),
    nn.ReLU(),
    nn.Linear(256, len(LABELS)),
)
state_dict = torch.load('pytorch_model.bin', map_location='cpu')
model.load_state_dict(state_dict, strict=False)
model.eval()

def predict(im):
    # Get RGBA image from Gradio sketchpad
    im = im["composite"]  # shape (H, W, 4)

    # Ensure it's a NumPy array
    if not isinstance(im, np.ndarray):
        im = np.array(im)

    # Convert RGBA to grayscale
    im = im[:, :, :3]  # Drop alpha
    im = np.dot(im, [0.299, 0.587, 0.114])  # Convert to grayscale (H, W)

    # Resize to 28x28 (what your model expects)
    im = Image.fromarray(im.astype(np.uint8)).resize((28, 28))
    im = np.array(im)

    # Normalize and convert to tensor
    x = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0  # shape [1, 1, 28, 28]

    with torch.no_grad():
        out = model(x)

    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    values, indices = torch.topk(probabilities, 5)

    return {LABELS[i]: v.item() for i, v in zip(indices, values)}

interface = gr.Interface(
    predict, 
    inputs="sketchpad", 
    outputs='label', 
    theme="huggingface", 
    title="Sketch Recognition", 
    description="Who wants to play Pictionary? Draw a common object like a shovel or a laptop, and the algorithm will guess in real time!", 
    article = "<p style='text-align: center'>Sketch Recognition | Demo Model</p>",
    live=True)
interface.launch(debug=True)
