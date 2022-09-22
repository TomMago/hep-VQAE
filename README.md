# Quantum Variational Autoencoders for HEP Analysis at the LHC

The code for the [QAE project](https://summerofcode.withgoogle.com/programs/2022/projects/ePnjKlJs). The original repo can be found [here](https://github.com/TomMago/hep-VQAE). In addition, more details can be found in my [post about the project](https://www.tommago.com).

## Setup

The most convenient way to set up the code is to create an environment

``` shell
conda create --name tfq
conda activate tfq
```

and install the requirement with

``` shell
pip install -r requirements.txt
```

Afterwards you can install the hep_VQAE package in development mode (execute in in repo root)
``` shell
python -m pip install -e .
```

and import the kernel to jupyter 

``` shell
python -m ipykernel install --user --name tfq
```
    
## Project

The goal of the project was to develop a Quantum Autoencoder (QAE) for anomaly detection in LHC data. Such an Autoencoder could be trained on Standard Model (SM) data and try to spot anomalous signals of Beyond the Standard Model (BSM) physics in datasets produced at the LHC.

### Datasets

Throughout the project I worked with different datasets:

#### MNIST

I used MNIST images for a first validation of ideas and debugging code samples.

#### Electron Photon 

The Electron Photon dataset contrain 32x32 ECAL images of electrons and photons.

<img src="assets/gammae.png">

#### Quark Gluon

The Quark Gluon dataset was my main object of study. For the most part, I rescaled the data to 12x12 in order to be able to simulate the demanding quantum algorithms. The original dataset contains a tracks, ECAL and HCAL channel, however for simplicity I only focus on the ECAL channel.

<img src="assets/quarkgluon.png">

### Architectures

In my project I implemented and tried many different architectures.
However my main focus was on the two following:

#### Fully Quantum Autoencoder

The fully Quantum Autoencoder (I abbreviate it as SQAE - Simple Quantum AutoEncoder) is based on [[1]](#References) and [[2]](#References).
The SQAE is structured as follows:

<img src="assets/qae.png">

The classical data is encoded with some unitarity and a parametrized unitarity is applied.
A SWAP-test computes the fidelity between the non-latent qubits and some trash qubits, which is then measured at the readout bit.

The SQAE is trained by maximizing the fidelity between non-latent qubits and the trash qubits. Because the transformation $U$ is unitary the decoder is simply the adjoint of the encoder. A more detailed description why the fidelity with the trash qubits can be used as loss can be found [here](https://www.tommago.com/posts/qae/). 

#### Hybrid Quantum Autoencoder

As a second approach I wanted to try a hybrid architecture, to combine QML with the strength of classical Deep learning. In particular I wanted to use classical CNNs to reduce the dimension of the data before feeding it into a PQC. Schematically the Hybrid Autoencoder (HAE) looks like this:

<img src="assets/hae.png">

The HAE is optimized using a classical reconstruction loss.

## Code structure

The notebooks expect the respective data to be stored in the data directory.

The main code can be found in the package hep_VQAE. There are implementations for the purely quantum autoencoder in pennylane and tf-quantum.
Furhtermore there are classical and hybrid models as well as auxiliary functions for data preprocessing and evaluation.

The notebooks folder contain example applications of the different models to the datasets. These notebooks are commented to walk you through the training.

The dev_notebooks folder on the other hand contains a lot of code and experiments I conducted throughout the project. There are experiments with classical Autoencoders, Jax implementations, vision transformers, different experiments with quantum convolutional circuits and much more. However this code is less documented. However I wanted to include it, maybe there is something someone might find useful in the future.


# References

[1] Romero, J., Olson, J. P., & Aspuru-Guzik, A. (2017). Quantum autoencoders for efficient compression of quantum data. Quantum Science and Technology, 2(4), 045001.

[2] Ngairangbam, V. S., Spannowsky, M., & Takeuchi, M. (2022). Anomaly detection in high-energy physics using a quantum autoencoder. Physical Review D, 105(9), 095004.


