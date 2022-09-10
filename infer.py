import os
import json
from io import BytesIO

from typing import Any
from cog import BasePredictor, Input, Path

import requests
from PIL import Image
from timm.models import create_model
from torch.nn.functional import softmax
from torchvision import transforms


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        
        self.labels2class = json.load(open("imagenet1000_labels.json", "r"))
        self.augs = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        self.model = create_model(
            "efficientnet_b3a", pretrained=True)

    # Define the arguments and types the model takes as input
    def predict(self,image: Path = Input(description="Image to classify")) -> Any:
        """Run a single prediction on the model"""

       
        img = Image.open(image).convert("RGB")
        
        # Preprocess the image
        img = self.augs(img).unsqueeze(0)
        out = self.model(img)
        out = softmax(out, dim=1)
        idx = out.argmax().item()

        return {"predicted": self.labels2class[str(idx)], "confidence": out[0,idx].item()}


