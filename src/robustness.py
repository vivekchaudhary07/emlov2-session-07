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
from captum.robust import FGSM
from omegaconf import DictConfig
from PIL import Image


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="robust.yaml")
def main(cfg: DictConfig):
    print(cfg)

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

    transform = T.Compose([T.Resize(cfg.imput_im_size), T.ToTensor(), T.Normalize(mean=cfg.MEAN, std=cfg.STD)])
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

        return predicted_label, prediction_score.squeeze().item(), pred_label_idx.item()

    for image_path in sources:
        print("=========:: ", image_path)
        out_path = os.path.join(cfg.results_dir, os.path.basename(image_path))
        os.makedirs(out_path, exist_ok=True)

        img = Image.open(image_path).convert("RGB")
        transformed_img = transform(img).unsqueeze(0).to(device)
        # img_tensor = transform_normalize(transformed_img).unsqueeze(0).to(device)
        pred_label, score, idx = get_prediction(model, transformed_img)

        for aug_name, func in cfg.augs.items():
            print(aug_name)
            if aug_name=='FGSM':
                fgsm = FGSM(model, lower_bound=-1, upper_bound=1)
                transformed_aug_img = fgsm.perturb(transformed_img, epsilon=0.16, target=idx) 
            else:
                aug = hydra.utils.instantiate(func)
                augmented_image = aug(image=np.array(img))['image']
                augmented_image = Image.fromarray(augmented_image)
                transformed_aug_img = transform(augmented_image).unsqueeze(0).to(device)
            
            aug_pred_label, aug_score, _ = get_prediction(model, transformed_aug_img)
            print(pred_label, score, '|', aug_pred_label, aug_score)

            if aug_name=='FGSM':
                augmented_image = inv_transform(transformed_aug_img).squeeze().permute(1, 2, 0).detach().numpy()

            plt.subplot(1, 2, 1)
            plt.imshow(np.array(img))
            plt.axis('off')
            plt.title('Original:\n%s | %s' %(pred_label, '%.4f' % score))

            plt.subplot(1, 2, 2)
            plt.imshow(np.array(augmented_image))
            plt.axis('off')
            plt.title('augmented:\n%s | %s' %(aug_pred_label, '%.4f' % aug_score))

            plt.suptitle(aug_name)
            plt.tight_layout()
            plt.savefig(os.path.join(out_path, aug_name + '.png'))


if __name__ == "__main__":
    main()
