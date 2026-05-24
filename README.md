# Image Generation

## Rhetorical Design

### Purpose

All image synthesis produces a discrete pixel grid. What differs is what that grid represents. In [cpu-raytracer](https://github.com/GalMunGral/cpu-raytracer) and [gl-raytracer](https://github.com/GalMunGral/gl-raytracer), each pixel estimates an integral of the radiance field over incoming light directions — the scene is a continuous physical model, and the image is a discrete approximation of it. In [gl-gaussian-splat](https://github.com/GalMunGral/gl-gaussian-splat), the scene is learned from photographs rather than specified, but the relationship is unchanged: the pixel grid remains a discretization of a continuous representation.

These notebooks take a different view. A model learns the joint distribution $`p(x)`$ over natural images from data, and generation is sampling from it. The pixel grid is not an approximation of a continuous physical signal — it is a draw from a learned distribution. There is no scene, and the question of what the image represents disappears with it.

### Strategy

Two paradigms are implemented side by side, both trained on CIFAR-10. The autoregressive model makes $`p(x)`$ explicit as a product of conditionals over discrete tokens; the diffusion model defines it implicitly through a learned reverse process. Placing them together makes the point that there is more than one way to formulate the same problem.