### Project Aurelius

I use a Roman Emperor as a codename for all my projects.
The last Emperor of the Pax Romana, the Emperor in an age of peace and stability in the Roman Empire, Marcus Aurelius would be happy to know I'm now using his name for a project to detect abrupt or anomalous changes in time series signals :)

## Background

A Varational Long Short Term Memory (LSTM) Autoencoder (src/model.py) was built using PyTorch. An Autoencoder is made of two models, an Encoder and a Decoder. The Encoder accepts a fixed length time series signal, and passes the signal through LSTM cells. The result is a vector representing the signal's temporal relationships. The *varational* in a Varational Autoencoder means there is one extra step: the vector is broken down further into its **mean** and **variance** vectors. The Decoder is just the reverse of the Encoder, its job is to use the mean and variance of the vector and reconstruct the original signal.

Interestingly we use Dynamic Time Warping (DTW) as a loss function. DTW is a similiarity metric between two time series signals, but its not as fast as using mean squared error. Part of the reason I wanted to do this project was to see if DTW gets better results than MSE. And using Cuda on my GTX 1660 Super, this runs really fast!

# The Goal

The model was trained on random samples of the function sin(f\*x + p). After training, if we pass in more samples from our sin function within our frequency limit we should expect a low reconstruction error. If change the signal, e.g., non-sinusodial, we should see a higher error. 

## See notebooks/tutorial.ipynb