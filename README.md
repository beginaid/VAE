# VAE
<div align="center">
    <img
        src="image\latent_space_plot.png"
        alt="Latent space of the test-set data."
        title="Latent space of the test-set data."
        width=500px>
    <figcaption>Mappings of the latent space in the test-set data.</figcaption>
</div>
This repository is for sharing the scripts of VAE.


# Description
You can train MNIST with any number of latent variable dimensions, and
visualize the latent space from multiple perspectives.


# Usage
You can use the following command to train and visualize VAE.
```
python main.py main --z_dim [The dimensions of latent variables]
```

You can start tensorboard with the following command.
Note that The default URL is http://localhost:6006/.
```
tensorboard --logdir ./logs
```

# Network
The configuration of VAE is shown as below.
<div align="center">
    <img
        src="image\network.png"
        alt="The configuration of VAE."
        title="The configuration of VAE."
        width=500px>
    <figcaption>The configuration of VAE.</figcaption>
</div>

# Output
You can get the following images.


## Reconstruction
<div align="center">
    <img
        src="image\reconstructions.png"
        alt="Reconstructions of the test-set data."
        title="Reconstructions of the test-set data."
        width=500px>
    <figcaption>Reconstructions of the test-set data.</figcaption>
</div>


## Mappings of the latent space
<div align="center">
    <img
        src="image\latent_space_scatter.png"
        alt="Mappings of the latent space in the test-set data."
        title="Mappings of the latent space in the test-set data."
        width=500px>
    <figcaption>Mappings of the latent space in the test-set data.</figcaption>
</div>

## Generation from lattice points
<div align="center">
    <img
        src="image\lattice_point.png"
        alt="Artificially generated lattice points."
        title="Artificially generated lattice points."
        width=500px>
    <figcaption>Artificially generated lattice points.</figcaption>
</div>
<br>
<div align="center">
    <img
        src="image\lattice_point_reconstruction.png"
        alt="Reconstructions generated from lattice points."
        title="Reconstructions generated from lattice points."
        width=500px>
    <figcaption>Reconstructions generated from lattice points.</figcaption>
</div>

## Walkthrough
<div align="center">
    <img
        src="image\latent_space_walking.png"
        alt="Four-direction-walkings at a constant speed in potential space."
        title="Four-direction-walkings at a constant speed in potential space."
        width=500px>
    <figcaption>Four-direction-walkings at a constant speed in potential space.</figcaption>
</div>
<br>
<div align="center">
    <img
        src="image\linear_changes.gif"
        alt="Reconstructions obtained from a walkthrough of the latent space."
        title="Reconstructions obtained from a walkthrough of the latent space."
        width=500px>
    <figcaption>Reconstructions obtained from a walkthrough of the latent space.</figcaption>
</div>
