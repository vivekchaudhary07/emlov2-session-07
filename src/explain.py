import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import urllib

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from omegaconf import DictConfig
from PIL import Image


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="explain.yaml")
def main(cfg: DictConfig):
    print(cfg)
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    img = Image.open(cfg.image_path)
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model_explain = hydra.utils.instantiate(cfg.explainability)

    img_tensor = transform(img).unsqueeze(0)
    
    output = model(img_tensor)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    print(type(pred_label_idx))
    output = model_explain(model=model, im=img_tensor, pred_label_idx=pred_label_idx)
    print(type(output))
    print(output)
    print(output.shape)

if __name__ == "__main__":
    main()

