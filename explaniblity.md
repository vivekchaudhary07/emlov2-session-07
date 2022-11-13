# TOC

- [TOC](#toc)
- [Assignment](#assignment)
- [Model Explanationability](#model-explanationability)
  - [Integrated Gradients](#integrated-gradients)
  - [Integrated Gradients with Noise](#integrated-gradients-with-noise)
  - [Saliency](#saliency)
  - [Occlusion](#occlusion)
  - [SHAP](#shap)
  - [GradCAM](#gradcam)
  - [GradCAM ++](#gradcam-)
- [Adversial Attacks wiht PGD](#adversial-attacks-wiht-pgd)
- [Model Robustness](#model-robustness)
# Assignment

1. Use Pretrained Models from TIMM (take models with larger input)
2. Do ALL the following for any 10 images taken by you (must be a class from ImageNet)
   1. Model Explanation with
      1. IG
      2. IG w/ Noise Tunnel
      3. Saliency
      4. Occlusion
      5. SHAP
      6. GradCAM
      7. GradCAM++
   2. Use PGD to make the model predict cat for all images
      1. save the images that made it predict cat
      2. add these images to the markdown file in your github repository
   3. Model Robustness with
      1. Pixel Dropout
      2. FGSM
      3. Random Noise
      4. Random Brightness 
    HINT: you can use https://albumentations.ai/ .for more image perturbations
3. Integrate above things into your pytorch lightning template
   1. create explain.py that will do all the model explanations
   2. create robustness.py to check for model robustness
4. Create a EXPLAINABILITY.md in log book folder of your repository
   1. Add the results (plots) of all the above things you’ve done
5. Submit link to EXPLAINABILITY.md in your github repository


` python src/explain.py source=images/test_images/ explainability=occlusion`

`python src/attacker.py source=images/test_images/`

`python src/robustness.py source=images/test_images/`

# Model Explanationability

![](images/1-model-explainability.png)

we will use algorithms in [captum](https://captum.ai/docs/attribution_algorithms)

<h3>examples images (these are my dogs 😁)</h3>

<p float="left">
    <img src="images/test_images/chair.jpg" width="200" />
    <img src="images/test_images/cup.png" width="200" />
    <img src="images/test_images/gsd_pup.jpg" width="200" />
    <img src="images/test_images/ludo.jpg" width="200" />
    <img src="images/test_images/ludo2.jpg" width="200" />
    <img src="images/test_images/ludo3.jpg" width="200" />
    <img src="images/test_images/oscar.jpg" width="200" />
    <img src="images/test_images/oscar2.jpg" width="200" />
    <img src="images/test_images/oscar3.jpg" width="200" />
    <img src="images/test_images/tom.jpeg" width="200" />
    <img src="images/test_images/tom2.jpg" width="200" />
    <img src="images/test_images/tom3.jpg" width="200" />
    <img src="images/test_images/tom4.jpg" width="200" />
    <img src="images/test_images/tom5.jpg" width="200" />
    <img src="images/test_images/tom6.jpg" width="200" />
</p>

## Integrated Gradients

-   https://captum.ai/docs/extension/integrated_gradients

<p float="left">
    <img src="images/modelexplainablity_outs/chair.jpg/ig.png" width="300" />
    <img src="images/modelexplainablity_outs/cup.png/ig.png" width="300" />
    <img src="images/modelexplainablity_outs/gsd_pup.jpg/ig.png" width="300" />
    <img src="images/modelexplainablity_outs/ludo.jpg/ig.png" width="300" />
    <img src="images/modelexplainablity_outs/ludo2.jpg/ig.png" width="300" />
    <img src="images/modelexplainablity_outs/ludo3.jpg/ig.png" width="300" />
    <img src="images/modelexplainablity_outs/oscar.jpg/ig.png" width="300" />
    <img src="images/modelexplainablity_outs/oscar2.jpg/ig.png" width="300" />
    <img src="images/modelexplainablity_outs/oscar3.jpg/ig.png" width="300" />
    <img src="images/modelexplainablity_outs/tom.jpeg/ig.png" width="300" />
    <img src="images/modelexplainablity_outs/tom2.jpg/ig.png" width="300" />
    <img src="images/modelexplainablity_outs/tom3.jpg/ig.png" width="300" />
    <img src="images/modelexplainablity_outs/tom4.jpg/ig.png" width="300" />
    <img src="images/modelexplainablity_outs/tom5.jpg/ig.png" width="300" />
    <img src="images/modelexplainablity_outs/tom6.jpg/ig.png" width="300" />
</p>


## Integrated Gradients with Noise

- https://captum.ai/api/noise_tunnel.html

<p float="left">
    <img src="images/modelexplainablity_outs/chair.jpg/igNoise.png" width="300" />
    <img src="images/modelexplainablity_outs/cup.png/igNoise.png" width="300" />
    <img src="images/modelexplainablity_outs/gsd_pup.jpg/igNoise.png" width="300" />
    <img src="images/modelexplainablity_outs/ludo.jpg/igNoise.png" width="300" />
    <img src="images/modelexplainablity_outs/ludo2.jpg/igNoise.png" width="300" />
    <img src="images/modelexplainablity_outs/ludo3.jpg/igNoise.png" width="300" />
    <img src="images/modelexplainablity_outs/oscar.jpg/igNoise.png" width="300" />
    <img src="images/modelexplainablity_outs/oscar2.jpg/igNoise.png" width="300" />
    <img src="images/modelexplainablity_outs/oscar3.jpg/igNoise.png" width="300" />
    <img src="images/modelexplainablity_outs/tom.jpeg/igNoise.png" width="300" />
    <img src="images/modelexplainablity_outs/tom2.jpg/igNoise.png" width="300" />
    <img src="images/modelexplainablity_outs/tom3.jpg/igNoise.png" width="300" />
    <img src="images/modelexplainablity_outs/tom4.jpg/igNoise.png" width="300" />
    <img src="images/modelexplainablity_outs/tom5.jpg/igNoise.png" width="300" />
    <img src="images/modelexplainablity_outs/tom6.jpg/igNoise.png" width="300" />
</p>


## Saliency

- https://captum.ai/api/saliency.html


<p float="left">
    <img src="images/modelexplainablity_outs/chair.jpg/saliency.jpg" width="300" />
    <img src="images/modelexplainablity_outs/cup.png/saliency.jpg" width="300" />
    <img src="images/modelexplainablity_outs/gsd_pup.jpg/saliency.jpg" width="300" />
    <img src="images/modelexplainablity_outs/ludo.jpg/saliency.jpg" width="300" />
    <img src="images/modelexplainablity_outs/ludo2.jpg/saliency.jpg" width="300" />
    <img src="images/modelexplainablity_outs/ludo3.jpg/saliency.jpg" width="300" />
    <img src="images/modelexplainablity_outs/oscar.jpg/saliency.jpg" width="300" />
    <img src="images/modelexplainablity_outs/oscar2.jpg/saliency.jpg" width="300" />
    <img src="images/modelexplainablity_outs/oscar3.jpg/saliency.jpg" width="300" />
    <img src="images/modelexplainablity_outs/tom.jpeg/saliency.jpg" width="300" />
    <img src="images/modelexplainablity_outs/tom2.jpg/saliency.jpg" width="300" />
    <img src="images/modelexplainablity_outs/tom3.jpg/saliency.jpg" width="300" />
    <img src="images/modelexplainablity_outs/tom4.jpg/saliency.jpg" width="300" />
    <img src="images/modelexplainablity_outs/tom5.jpg/saliency.jpg" width="300" />
    <img src="images/modelexplainablity_outs/tom6.jpg/saliency.jpg" width="300" />
</p>


## Occlusion

- https://captum.ai/api/occlusion.html

<p float="left">
    <img src="images/modelexplainablity_outs/chair.jpg/occlusion.jpg" width="300" />
    <img src="images/modelexplainablity_outs/cup.png/occlusion.jpg" width="300" />
    <img src="images/modelexplainablity_outs/gsd_pup.jpg/occlusion.jpg" width="300" />
    <img src="images/modelexplainablity_outs/ludo.jpg/occlusion.jpg" width="300" />
    <img src="images/modelexplainablity_outs/ludo2.jpg/occlusion.jpg" width="300" />
    <img src="images/modelexplainablity_outs/ludo3.jpg/occlusion.jpg" width="300" />
    <img src="images/modelexplainablity_outs/oscar.jpg/occlusion.jpg" width="300" />
    <img src="images/modelexplainablity_outs/oscar2.jpg/occlusion.jpg" width="300" />
    <img src="images/modelexplainablity_outs/oscar3.jpg/occlusion.jpg" width="300" />
    <img src="images/modelexplainablity_outs/tom.jpeg/occlusion.jpg" width="300" />
    <img src="images/modelexplainablity_outs/tom2.jpg/occlusion.jpg" width="300" />
    <img src="images/modelexplainablity_outs/tom3.jpg/occlusion.jpg" width="300" />
    <img src="images/modelexplainablity_outs/tom4.jpg/occlusion.jpg" width="300" />
    <img src="images/modelexplainablity_outs/tom5.jpg/occlusion.jpg" width="300" />
    <img src="images/modelexplainablity_outs/tom6.jpg/occlusion.jpg" width="300" />
</p>


## SHAP

- https://captum.ai/api/gradient_shap.html

<p float="left">
    <img src="images/modelexplainablity_outs/chair.jpg/shap.jpg" width="300" />
    <img src="images/modelexplainablity_outs/cup.png/shap.jpg" width="300" />
    <img src="images/modelexplainablity_outs/gsd_pup.jpg/shap.jpg" width="300" />
    <img src="images/modelexplainablity_outs/ludo.jpg/shap.jpg" width="300" />
    <img src="images/modelexplainablity_outs/ludo2.jpg/shap.jpg" width="300" />
    <img src="images/modelexplainablity_outs/ludo3.jpg/shap.jpg" width="300" />
    <img src="images/modelexplainablity_outs/oscar.jpg/shap.jpg" width="300" />
    <img src="images/modelexplainablity_outs/oscar2.jpg/shap.jpg" width="300" />
    <img src="images/modelexplainablity_outs/oscar3.jpg/shap.jpg" width="300" />
    <img src="images/modelexplainablity_outs/tom.jpeg/shap.jpg" width="300" />
    <img src="images/modelexplainablity_outs/tom2.jpg/shap.jpg" width="300" />
    <img src="images/modelexplainablity_outs/tom3.jpg/shap.jpg" width="300" />
    <img src="images/modelexplainablity_outs/tom4.jpg/shap.jpg" width="300" />
    <img src="images/modelexplainablity_outs/tom5.jpg/shap.jpg" width="300" />
    <img src="images/modelexplainablity_outs/tom6.jpg/shap.jpg" width="300" />
</p>


- https://github.com/jacobgil/pytorch-grad-cam
## GradCAM

<p float="left">
    <img src="images/modelexplainablity_outs/chair.jpg/cam.png" width="300" />
    <img src="images/modelexplainablity_outs/cup.png/cam.png" width="300" />
    <img src="images/modelexplainablity_outs/gsd_pup.jpg/cam.png" width="300" />
    <img src="images/modelexplainablity_outs/ludo.jpg/cam.png" width="300" />
    <img src="images/modelexplainablity_outs/ludo2.jpg/cam.png" width="300" />
    <img src="images/modelexplainablity_outs/ludo3.jpg/cam.png" width="300" />
    <img src="images/modelexplainablity_outs/oscar.jpg/cam.png" width="300" />
    <img src="images/modelexplainablity_outs/oscar2.jpg/cam.png" width="300" />
    <img src="images/modelexplainablity_outs/oscar3.jpg/cam.png" width="300" />
    <img src="images/modelexplainablity_outs/tom.jpeg/cam.png" width="300" />
    <img src="images/modelexplainablity_outs/tom2.jpg/cam.png" width="300" />
    <img src="images/modelexplainablity_outs/tom3.jpg/cam.png" width="300" />
    <img src="images/modelexplainablity_outs/tom4.jpg/cam.png" width="300" />
    <img src="images/modelexplainablity_outs/tom5.jpg/cam.png" width="300" />
    <img src="images/modelexplainablity_outs/tom6.jpg/cam.png" width="300" />
</p>

## GradCAM ++

<p float="left">
    <img src="images/modelexplainablity_outs/chair.jpg/cam++.png" width="300" />
    <img src="images/modelexplainablity_outs/cup.png/cam++.png" width="300" />
    <img src="images/modelexplainablity_outs/gsd_pup.jpg/cam++.png" width="300" />
    <img src="images/modelexplainablity_outs/ludo.jpg/cam++.png" width="300" />
    <img src="images/modelexplainablity_outs/ludo2.jpg/cam++.png" width="300" />
    <img src="images/modelexplainablity_outs/ludo3.jpg/cam++.png" width="300" />
    <img src="images/modelexplainablity_outs/oscar.jpg/cam++.png" width="300" />
    <img src="images/modelexplainablity_outs/oscar2.jpg/cam++.png" width="300" />
    <img src="images/modelexplainablity_outs/oscar3.jpg/cam++.png" width="300" />
    <img src="images/modelexplainablity_outs/tom.jpeg/cam++.png" width="300" />
    <img src="images/modelexplainablity_outs/tom2.jpg/cam++.png" width="300" />
    <img src="images/modelexplainablity_outs/tom3.jpg/cam++.png" width="300" />
    <img src="images/modelexplainablity_outs/tom4.jpg/cam++.png" width="300" />
    <img src="images/modelexplainablity_outs/tom5.jpg/cam++.png" width="300" />
    <img src="images/modelexplainablity_outs/tom6.jpg/cam++.png" width="300" />
</p>

# Adversial Attacks wiht PGD

we will use [pgd](https://arxiv.org/abs/1706.06083) to predict every class as tiger cat

- https://captum.ai/api/robust.html

<p float="left">
    <img src="images/adversarial_attacks/chair.jpg" width="300" />
    <img src="images/adversarial_attacks/cup.png" width="300" />
    <img src="images/adversarial_attacks/gsd_pup.jpg" width="300" />
    <img src="images/adversarial_attacks/ludo.jpg" width="300" />
    <img src="images/adversarial_attacks/ludo2.jpg" width="300" />
    <img src="images/adversarial_attacks/ludo3.jpg" width="300" />
    <img src="images/adversarial_attacks/oscar.jpg" width="300" />
    <img src="images/adversarial_attacks/oscar2.jpg" width="300" />
    <img src="images/adversarial_attacks/oscar3.jpg" width="300" />
    <img src="images/adversarial_attacks/tom.jpeg" width="300" />
    <img src="images/adversarial_attacks/tom2.jpg" width="300" />
    <img src="images/adversarial_attacks/tom3.jpg" width="300" />
    <img src="images/adversarial_attacks/tom4.jpg" width="300" />
    <img src="images/adversarial_attacks/tom5.jpg" width="300" />
    <img src="images/adversarial_attacks/tom6.jpg" width="300" />
</p>


# Model Robustness


<p float="left">
    <img src="images/robust/chair.jpg/FGSM.png" width="600" />
    <img src="images/robust/chair.jpg/gaussian_noise.png" width="600" />
    <img src="images/robust/chair.jpg/pixel_dropout.png" width="600" />
    <img src="images/robust/chair.jpg/random_brightness.png" width="600" />
</p>



<p float="left">
    <img src="images/robust/cup.png/FGSM.png" width="600" />
    <img src="images/robust/cup.png/gaussian_noise.png" width="600" />
    <img src="images/robust/cup.png/pixel_dropout.png" width="600" />
    <img src="images/robust/cup.png/random_brightness.png" width="600" />
</p>


<p float="left">
    <img src="images/robust/gsd_pup.jpg/FGSM.png" width="600" />
    <img src="images/robust/gsd_pup.jpg/gaussian_noise.png" width="600" />
    <img src="images/robust/gsd_pup.jpg/pixel_dropout.png" width="600" />
    <img src="images/robust/gsd_pup.jpg/random_brightness.png" width="600" />
</p>


<p float="left">
    <img src="images/robust/ludo.jpg/FGSM.png" width="600" />
    <img src="images/robust/ludo.jpg/gaussian_noise.png" width="600" />
    <img src="images/robust/ludo.jpg/pixel_dropout.png" width="600" />
    <img src="images/robust/ludo.jpg/random_brightness.png" width="600" />
</p>


<p float="left">
    <img src="images/robust/ludo2.jpg/FGSM.png" width="600" />
    <img src="images/robust/ludo2.jpg/gaussian_noise.png" width="600" />
    <img src="images/robust/ludo2.jpg/pixel_dropout.png" width="600" />
    <img src="images/robust/ludo2.jpg/random_brightness.png" width="600" />
</p>


<p float="left">
    <img src="images/robust/ludo3.jpg/FGSM.png" width="600" />
    <img src="images/robust/ludo3.jpg/gaussian_noise.png" width="600" />
    <img src="images/robust/ludo3.jpg/pixel_dropout.png" width="600" />
    <img src="images/robust/ludo3.jpg/random_brightness.png" width="600" />
</p>


<p float="left">
    <img src="images/robust/oscar.jpg/FGSM.png" width="600" />
    <img src="images/robust/oscar.jpg/gaussian_noise.png" width="600" />
    <img src="images/robust/oscar.jpg/pixel_dropout.png" width="600" />
    <img src="images/robust/oscar.jpg/random_brightness.png" width="600" />
</p>


<p float="left">
    <img src="images/robust/oscar3.jpg/FGSM.png" width="600" />
    <img src="images/robust/oscar3.jpg/gaussian_noise.png" width="600" />
    <img src="images/robust/oscar3.jpg/pixel_dropout.png" width="600" />
    <img src="images/robust/oscar3.jpg/random_brightness.png" width="600" />
</p>

<p float="left">
    <img src="images/robust/tom.jpeg/FGSM.png" width="600" />
    <img src="images/robust/tom.jpeg/gaussian_noise.png" width="600" />
    <img src="images/robust/tom.jpeg/pixel_dropout.png" width="600" />
    <img src="images/robust/tom.jpeg/random_brightness.png" width="600" />
</p>

<p float="left">
    <img src="images/robust/tom2.jpg/FGSM.png" width="600" />
    <img src="images/robust/tom2.jpg/gaussian_noise.png" width="600" />
    <img src="images/robust/tom2.jpg/pixel_dropout.png" width="600" />
    <img src="images/robust/tom2.jpg/random_brightness.png" width="600" />
</p>

<p float="left">
    <img src="images/robust/tom3.jpg/FGSM.png" width="600" />
    <img src="images/robust/tom3.jpg/gaussian_noise.png" width="600" />
    <img src="images/robust/tom3.jpg/pixel_dropout.png" width="600" />
    <img src="images/robust/tom3.jpg/random_brightness.png" width="600" />
</p>

<p float="left">
    <img src="images/robust/tom4.jpg/FGSM.png" width="600" />
    <img src="images/robust/tom4.jpg/gaussian_noise.png" width="600" />
    <img src="images/robust/tom4.jpg/pixel_dropout.png" width="600" />
    <img src="images/robust/tom4.jpg/random_brightness.png" width="600" />
</p>

<p float="left">
    <img src="images/robust/tom5.jpg/FGSM.png" width="600" />
    <img src="images/robust/tom5.jpg/gaussian_noise.png" width="600" />
    <img src="images/robust/tom5.jpg/pixel_dropout.png" width="600" />
    <img src="images/robust/tom5.jpg/random_brightness.png" width="600" />
</p>

<p float="left">
    <img src="images/robust/tom6.jpg/FGSM.png" width="600" />
    <img src="images/robust/tom6.jpg/gaussian_noise.png" width="600" />
    <img src="images/robust/tom6.jpg/pixel_dropout.png" width="600" />
    <img src="images/robust/tom6.jpg/random_brightness.png" width="600" />
</p>

