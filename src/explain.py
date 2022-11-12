import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
from glob import glob

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from omegaconf import DictConfig
from PIL import Image
from pytorch_lightning import LightningModule


torch.manual_seed(0)
np.random.seed(0)


@hydra.main(
    version_base="1.2", config_path=root / "configs", config_name="explain.yaml"
)
def main(cfg: DictConfig):
    print(cfg)

    sources = [cfg.source] if os.path.isfile(cfg.source) else glob(f"{cfg.source}/*")
    print(len(sources))
    for image_path in sources:
        print(image_path)
        out_path = os.path.join("images/modelexplainablity_outs", os.path.basename(image_path))
        os.makedirs(out_path, exist_ok=True)

        device = torch.device("cuda")

        transform = T.Compose([T.Resize(cfg.imput_im_size), T.ToTensor()])
        transform_normalize = T.Normalize(mean=cfg.MEAN, std=cfg.STD)

        model: LightningModule = hydra.utils.instantiate(cfg.model)
        model = model.to(device)
        model_explain = hydra.utils.instantiate(cfg.explainability)

        img = Image.open(image_path)
        transformed_img = transform(img)
        img_tensor = transform_normalize(transformed_img).unsqueeze(0).to(device)

        output = model(img_tensor)
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)
        # print(type(pred_label_idx))

        model_explain(
            model=model,
            img_tensor=img_tensor,
            np_img=np.transpose(transformed_img.cpu().detach().numpy(), (1, 2, 0)),
            pred_label_idx=pred_label_idx,
            save_path=out_path,
        )


if __name__ == "__main__":
    main()
