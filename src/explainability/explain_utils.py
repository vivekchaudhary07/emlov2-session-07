import os

import numpy as np
import torch
from captum.attr import (
    IntegratedGradients,
    NoiseTunnel,
    Occlusion,
    Saliency,
    GradientShap,
)
from captum.attr import visualization as viz


def integratedgradients(
    model: torch.nn.Module,
    img_tensor: torch.tensor,
    np_img: np.array,
    pred_label_idx: torch.tensor,
    save_path: str,
    label,
    color_map,
):
    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(
        img_tensor, target=pred_label_idx, n_steps=200
    )

    out = viz.visualize_image_attr(
        np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np_img,
        method="heat_map",
        cmap=color_map,
        show_colorbar=True,
        title=label,
        sign="positive",
        outlier_perc=1,
    )
    out[0].savefig(os.path.join(save_path, "ig.png"))


def ig_noise(
    model: torch.nn.Module,
    img_tensor: torch.tensor,
    np_img: np.array,
    pred_label_idx: torch.tensor,
    save_path: str,
    label,
    color_map,
):
    integrated_gradients = IntegratedGradients(model)
    noise_tunnel = NoiseTunnel(integrated_gradients)

    attributions_ig_nt = noise_tunnel.attribute(
        img_tensor, nt_samples=10, nt_type="smoothgrad_sq", target=pred_label_idx
    )

    out = viz.visualize_image_attr(
        np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np_img,
        method="heat_map",
        cmap=color_map,
        show_colorbar=True,
        title=label,
        sign="positive",
        outlier_perc=1,
    )
    out[0].savefig(os.path.join(save_path, "igNoise.png"))


def occlusion(
    model: torch.nn.Module,
    img_tensor: torch.tensor,
    np_img: np.array,
    pred_label_idx: torch.tensor,
    save_path: str,
    label,
    color_map,
):
    occlusion_ = Occlusion(model)

    attributions_occ = occlusion_.attribute(
        img_tensor,
        strides=(3, 8, 8),
        target=pred_label_idx,
        sliding_window_shapes=(3, 15, 15),
        baselines=0,
    )

    out = viz.visualize_image_attr(
        np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np_img,
        method="heat_map",
        # cmap=color_map,
        show_colorbar=True,
        title=label,
        sign="positive",
        outlier_perc=2,
    )
    out[0].savefig(os.path.join(save_path, "occlusion.jpg"))


def saliency(
    model: torch.nn.Module,
    img_tensor: torch.tensor,
    np_img: np.array,
    pred_label_idx: torch.tensor,
    save_path: str,
    label,
):
    # img_tensor.requires_grad

    saliency_ = Saliency(model)
    grads = saliency_.attribute(img_tensor, target=pred_label_idx.item())

    out = viz.visualize_image_attr(
        np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np_img,
        method="blended_heat_map",
        show_colorbar=True,
        title=label,
        sign="absolute_value",
        outlier_perc=1,
    )
    out[0].savefig(os.path.join(save_path, "saliency.jpg"))


def grad_shap(
    model: torch.nn.Module,
    img_tensor: torch.tensor,
    np_img: np.array,
    pred_label_idx: torch.tensor,
    save_path: str,
    label,
    color_map,
):

    # Defining baseline distribution of images
    rand_img_dist = torch.cat([img_tensor * 0, img_tensor * 1])

    gradient_shap = GradientShap(model)
    attributions_gs = gradient_shap.attribute(
        img_tensor,
        n_samples=50,
        stdevs=0.0001,
        baselines=rand_img_dist,
        target=pred_label_idx,
    )

    out = viz.visualize_image_attr(
        np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np_img,
        method="heat_map",
        show_colorbar=True,
        title=label,
        sign="absolute_value",
        cmap=color_map,
    )
    out[0].savefig(os.path.join(save_path, "shap.jpg"))
