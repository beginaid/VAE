# VAE
This repository is for sharing the scripts of VAE.
<p align="center">
    <img
        src="image\latent_space_plot.png"
        alt="Latent space of the test-set data."
        title="Latent space of the test-set data."
        width=500px>
</p>


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
<p align="center">
    <img
        src="image\network.png"
        alt="The configuration of VAE."
        title="The configuration of VAE."
        width=500px>
</p>

# Output
You can get the following images.


## Reconstruction
<p align="center">
    <img
        src="image\reconstructions.png"
        alt="Reconstructions of the test-set data."
        title="Reconstructions of the test-set data."
        width=500px>
</p>


## Mappings of the latent space
<p align="center">
    <img
        src="image\latent_space_scatter.png"
        alt="Mappings of the latent space in the test-set data."
        title="Mappings of the latent space in the test-set data."
        width=500px>
</p>

## Generation from lattice points
<p align="center">
    <img
        src="image\lattice_point.png"
        alt="Artificially generated lattice points."
        title="Artificially generated lattice points."
        width=500px>
</p>
<br>
<p align="center">
    <img
        src="image\lattice_point_reconstruction.png"
        alt="Reconstructions generated from lattice points."
        title="Reconstructions generated from lattice points."
        width=500px>
</p>

## Walkthrough
<p align="center">
    <img
        src="image\latent_space_walking.png"
        alt="Four-direction-walkings at a constant speed in potential space."
        title="Four-direction-walkings at a constant speed in potential space."
        width=500px>
</p>
<br>
<p align="center">
    <img
        src="image\linear_changes.gif"
        alt="Reconstructions obtained from a walkthrough of the latent space."
        title="Reconstructions obtained from a walkthrough of the latent space."
        width=500px>
</p>
