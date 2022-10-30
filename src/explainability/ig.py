import torch

from captum.attr import IntegratedGradients
from captum.attr import visualization as viz


def integratedgradients(model: torch.nn.Module, im: torch.tensor, pred_label_idx: torch.tensor, color_map):
    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(im, target=pred_label_idx, n_steps=200)

    out = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                                np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                method='heat_map',
                                cmap=color_map,
                                show_colorbar=True,
                                sign='positive',
                                outlier_perc=1)

    return out
