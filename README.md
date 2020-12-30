### Project Aurelius

As the last Emperor of the Pax Romana, the age of peace and stability in the Roman Empire, Marcus Aurelius would be happy to know I'm now using his name for a project to detect abrupt or anomalous changes in time series signals.

## Background

A Varational Long Short Term Memory (LSTM) Autoencoder (src/model.py) was built using PyTorch. An Autoencoder is made of two models, an Encoder and a Decoder. The Encoder accepts a fixed length time series signal, and passes the signal through LSTM cells. The result is a vector representing the signal's temporal relationships. The *varational* in a Varational Autoencoder means there is one extra step: the vector is broken down further into its **mean** and **variance** vectors. The Decoder is just the reverse of the Encoder, its job is to use the mean and variance of the vector and reconstruct the original signal.

# The Goal

The model was trained on 10,000 random samples of the function sin(f\*x + p) where the phase p is 0 < p < 2 \* pi, and the frequency f is 0 < f < 0.1. After training, if we pass in more samples from our sin function within our frequency limit we should expect a low reconstruction error. If change the signal, e.g., we increase the frequency to past our limit of 0.1, we should see a higher error. 

## Why
A heat exchanger is a system to transfer heat between two fluids. Online monitoring of commercial heat exchangers is done by tracking the overall heat transfer coefficient. Fouling, caused by the build-up of impurities in the heat exhanger, increases the resistance to heat transfer, thereby decreasing the overall heat transfer coefficient. But some studies have shown that fouling can actually increase the heat transfer coefficient for a short period of time thanks to greater roughness on the heat transfer surface. Either way, fouling definitely causes the heat transfer coefficient to act abnormally, which is what we want. A simple way to detect fouling is if the heat transfer coefficient dips below a threshold. More interestingly, can the autoencoder detect a pattern change caused by the impurity build-up far before the threshold is broken. If yes, then the Autoencoder method can save owner operators money in maintenance costs thanks to early detection of fouling. 
## See notebooks/tutorial.ipynb