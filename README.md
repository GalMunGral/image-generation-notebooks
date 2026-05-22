# Image Generation

## Rhetorical Design

### Purpose

All image synthesis produces a discrete pixel grid. What differs is what that grid represents. In [cpu-raytracer](https://github.com/GalMunGral/cpu-raytracer) and [gl-raytracer](https://github.com/GalMunGral/gl-raytracer), each pixel estimates an integral of the radiance field over incoming light directions — the scene is a continuous physical model, and the image is a discrete approximation of it. In [gl-gaussian-splat](https://github.com/GalMunGral/gl-gaussian-splat), the scene is learned from photographs rather than specified, but the relationship is unchanged: the pixel grid remains a discretization of a continuous representation.

These notebooks take a different view. A model learns the joint distribution $p(x)$ over natural images from data, and generation is sampling from it. The pixel grid is not an approximation of a continuous physical signal — it is a draw from a learned distribution. There is no scene, and the question of what the image represents disappears with it.

### Strategy

Two paradigms are implemented side by side, both trained on CIFAR-10. The autoregressive model makes $p(x)$ explicit as a product of conditionals over discrete tokens; the diffusion model defines it implicitly through a learned reverse process. Placing them together makes the point that there is more than one way to formulate the same problem.

## Technical Challenges

### VQ-VAE

Direct autoregressive modeling over raw pixels is intractable — a 32×32 RGB image has 3072 dimensions, and modeling each pixel conditional on all previous ones requires extreme model capacity. A VQ-VAE first compresses the image to a small grid of discrete tokens. The encoder maps $x$ to a continuous spatial latent $z_e \in \mathbb{R}^{d \times H' \times W'}$; the vector quantizer replaces each spatial position with the nearest entry in a learned codebook $\{e_k\}_{k=1}^K$:

```math
z_q = e_{\,\arg\min_k \|z_e - e_k\|}
```

Since `argmin` has no gradient, the decoder receives a straight-through copy: $z_q^{\text{st}} = z_e + (z_q - z_e).\texttt{detach()}$ — the forward pass uses $z_q$, the backward pass acts as if $z_e$ were used directly. The codebook is updated by exponential moving average. The training loss is

```math
\mathcal{L} = \|x - \hat{x}\|^2 + \beta\,\|z_q.\texttt{detach()} - z_e\|^2
```

### Autoregressive Transformer

With images compressed to token sequences, modeling $p(x)$ reduces to modeling $p(t_1, \ldots, t_N \mid c)$ where $c$ is the class label and each $t_i \in \{1, \ldots, K\}$ is a codebook index. A GPT-style transformer with causal self-attention factorizes this autoregressively:

```math
p(t_1, \ldots, t_N \mid c) = \prod_{i=1}^{N} p(t_i \mid c,\, t_1, \ldots, t_{i-1})
```

The label is prepended as the first input token. Since tokens lie on a 2D spatial grid, each flat index is decomposed into row and column, each encoded independently with sinusoidal embeddings and concatenated. The model is trained with cross-entropy loss; at generation time tokens are sampled one by one and decoded by the VQ-VAE decoder.

### Diffusion

The forward process gradually destroys an image by adding Gaussian noise over $T$ steps. The marginal at step $t$ has the closed form

```math
q(x_t \mid x_0) = \mathcal{N}\!\left(\sqrt{\bar{\alpha}_t}\,x_0,\;(1 - \bar{\alpha}_t)\mathbf{I}\right), \qquad \bar{\alpha}_t = \prod_{s=1}^{t}(1 - \beta_s)
```

which allows sampling $x_t$ directly from $x_0$ without simulating the chain. A U-Net $\varepsilon_\theta(x_t, t)$ is trained to predict the added noise, minimizing $\|\varepsilon - \varepsilon_\theta(x_t, t)\|^2$. Generation runs the reverse process: starting from $x_T \sim \mathcal{N}(0, \mathbf{I})$, each step samples from the posterior $q(x_{t-1} \mid x_t, x_0)$ — where $x_0$ is estimated from the predicted noise — whose mean and variance are available in closed form by Bayes' rule.