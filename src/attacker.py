import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
import urllib
from glob import glob

import hydra
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from captum.robust import PGD
from omegaconf import DictConfig
from PIL import Image


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="attack.yaml")
def main(cfg: DictConfig):
    print(cfg)
    os.makedirs(cfg.results_dir, exist_ok=True)

    # Download human-readable labels for ImageNet.
    # get the classnames
    url, filename = (
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        "imagenet_classes.txt",
    )
    urllib.request.urlretrieve(url, filename)
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    sources = [cfg.source] if os.path.isfile(cfg.source) else glob(f"{cfg.source}/*")
    print("Total images:: ", len(sources))

    device = torch.device(cfg.device)

    transform = T.Compose([T.Resize(cfg.imput_im_size), T.ToTensor()])
    transform_normalize = T.Normalize(mean=cfg.MEAN, std=cfg.STD)

    inv_transform = T.Compose(
        [
            T.Normalize(
                mean=(-1 * np.array(cfg.MEAN) / np.array(cfg.STD)).tolist(),
                std=(1 / np.array(cfg.STD)).tolist(),
            ),
        ]
    )
    model = hydra.utils.instantiate(cfg.model)
    model.eval()
    model = model.to(device)

    def get_prediction(model, image: torch.Tensor):
        model = model.to(device)
        img_tensor = image.to(device)
        with torch.no_grad():
            output = model(img_tensor)
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)

        pred_label_idx.squeeze_()
        predicted_label = categories[pred_label_idx.item()]

        return predicted_label, prediction_score.squeeze().item()

    for image_path in sources:
        print(image_path)

        img = Image.open(image_path).convert("RGB")
        transformed_img = transform(img)
        img_tensor = transform_normalize(transformed_img).unsqueeze(0).to(device)

        # construct the PGD attacker
        pgd = PGD(
            model,
            torch.nn.CrossEntropyLoss(reduction="none"),
            lower_bound=-1,
            upper_bound=1,
        )

        perturbed_image_pgd = pgd.perturb(
            inputs=img_tensor,
            radius=0.13,
            step_size=0.02,
            step_num=7,
            target=torch.tensor([cfg.target]).to(device),
            targeted=True,
        )

        new_pred_pgd, score_pgd = get_prediction(model, perturbed_image_pgd)
        npimg = (
            inv_transform(perturbed_image_pgd.cpu())
            .squeeze()
            .permute(1, 2, 0)
            .detach()
            .numpy()
        )
        plt.imshow(npimg)
        plt.title("prediction: %s" % new_pred_pgd + " " + str(score_pgd))
        plt.axis("off")
        plt.savefig(os.path.join(cfg.results_dir, os.path.basename(image_path)))


if __name__ == "__main__":
    main()
