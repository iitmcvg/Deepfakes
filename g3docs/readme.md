## Deep Fake ( Deep Learning + Fake ) is a human image synthesis technique using artificial intelligence methods. This project aims to

## develop a robust and fast face morphing system for images and video, using Autoencoders and GANs.

# Deepfakes

## CVI Group- Arnav Anil Mhaske, Anand Uday Gokhale

```
We aim to achieve the objective of the project by recreating the expressions
and position of a person’s face with the style and features of another person
using CNNs.
```

```
We have an encoder for all faces, and a decoder for every specific person. So, we train the
model to encode all faces into features and train the decoder to recreate the input image.
Now when we use the decoder of a different person on an encoded face, we get the
expressions of the encoded face with the features of the other person.
```
```
We use convolutions and max-pooling to downsample an RGB input image to a latent space.
Then we deconvolve and up-sample the encoded layer back into an RGB image which is
compared with the original while training.
```
```
We experimented with the following loss functions to compare the reconstructed image to the
input-
```
```
MSE - This is a pixel by pixel difference squared and added up to give a single loss number.
This works as a low pass filter and does not capture details, rather it tended to give an
average estimate of the person’s face which did not vary much with our inputs.
```
```
Perceptual Loss - This uses a pre-trained model, namely vgg16’s weights to produce 3 losses
: feature, style and tv loss. This worked better than MSE but still gave fairly blurry outputs.
This however did not average the faces as MSE did.
```
```
VAE : Variational Autoencoder’s didn’t reconstruct the image at all but instead gave an
output that looked like an average of all the faces that were a part of the training set.
```
```
GANs - Using GANs didn’t give us desired outputs. The discriminator trained way faster than
the generator and the outputs didn’t resemble faces. This was because the network wasn’t
deep enough. Our experiments to integrate GANs into this network still continue.
```
![results](https://github.com/iitmcvg/Deepfakes/blob/master/g3docs/reconstructed.jpeg)

```
CURRENT AND FUTURE WORK:
We are currently working on implementing deep video portraits. In this
we pass 3 inputs to an Autoencoder + GANs Model.Namely a 3D
head pose, eye gaze, and a face mask generated from the
input.Currently we have a working model of the 3d head-pose
generator.The results of this are shown below:
```
![3Dpose](https://github.com/iitmcvg/Deepfakes/blob/master/g3docs/reconstructed.jpeg)



