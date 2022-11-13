import os

import torch
import matplotlib.pyplot as plt
import numpy as np

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def grad_cam(
    model: torch.nn.Module,
    img_tensor: torch.tensor,
    np_img: np.array,
    pred_label_idx: torch.tensor,
    save_path: str,
    label,
):
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    targets = [ClassifierOutputTarget(pred_label_idx.item())]
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(np_img, grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.title(label)
    plt.axis("off")
    plt.savefig(os.path.join(save_path, "cam.png"))


def grad_cam_plusplus(
    model: torch.nn.Module,
    img_tensor: torch.tensor,
    np_img: np.array,
    pred_label_idx: torch.tensor,
    save_path: str,
    label,
):
    target_layers = [model.layer4[-1]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)

    targets = [ClassifierOutputTarget(pred_label_idx.item())]
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(np_img, grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.title(label)
    plt.axis("off")
    plt.savefig(os.path.join(save_path, "cam++.png"))
