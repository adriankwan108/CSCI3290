Requirement:    
<pre>
Python 3.4+     
numpy   
opencv
PyTorch 1.2+
</pre>

**Background:**     
<pre>
Image restoration is the operation of taking a noisy/corrupt image and estimating the clean, original image. 
The noise may come from motion blur, mis-focus, age, etc.,
the image could have some pixel missed or color changed.

The restoration can be performed by reversing the process that blurred the image, 
which means imaging the point source and use that point source image. (Point Spread Function).

One of the classical image restoration problems is single image super resolution (SISR).
It aims to recover high-resolution (HR) image from low-resolution (LR) image.
It could be used in a variety of applications,
such as medical imaging, security, and surveillance imaging.

There are several ways to solve the above problems, artifical work or algorithms,
e.g. prediction, edge based, image statistacl, patch based.
In modern technology, deep learning is a popular approcah.
This program will focus on using
a super resolution convolutional neural network (SRCNN) with PyTorch,
with “Learning a Deep Convolutional Network for Image Super-Resolution” [1] as the 
basic reference.

*Drawbacks: high computation cost due to bicubic interpolation and non-linear mapping step
            
            Some articles in the Internet said
            it fails to capture self-similarity in non-local patches
            and not robust to noise due to training and testing sets,
            but some research found SRCNN is robust when comparing to other deep learning method.
</pre>

**Reference**
<pre>
[1] Dong, C., Loy, C. C., He, K., & Tang, X. (2015). Image super-resolution using deep convolutional 
networks. IEEE transactions on pattern analysis and machine intelligence, 38(2), 295-307.
</pre>

**Usage:**    
<pre>
Restore LR image to HR image through SRCNN with PyTorch.
</pre>

**Algorithm:**          
<pre>
The SRCNN is a simple feed-forward neural network. It upscaled the input LR, feeds the upscaled image 
through several layers one after the other, and then finally gives the output. 
The overall training procedure of this network is the same as the above framework.

SRCNN:
1. Preprocessing: Upscales LR image to desired HR size (using bicubic interpolation).

2. Feature extraction: Extracts a set of feature maps from the upscaled LR image.
3. Non-linear mapping: Maps the feature maps representing LR to HR patches.
4. Reconstruction: Produces the HR image from HR patches

Operation 2~4 can be cast as a convolutional layer in a CNN, 
which accepts the upscaled images as input, and outputs the HR image.

To use SRCNN, image databases are created, containing LR-HR pairs and used as a training set.
Model Training:
Define the neural network that has some learnable parameters (or weights)
• Iterate over a dataset of inputs
• Process input through the network
• Compute the loss between output and the ground truth (how far is the output from being correct)
• Propagate gradients back into the network’s parameters
• Update the weights of the network, typically using a simple update rule: weight = weight -
learning_rate * gradient

The loss function in this programme is mean squared error provided by PyTorch.
Using MSE as the loss function favors a high peak signal-to-noise ratio (PSNR). 
The PSNR is a widely used metric for quantitatively evaluating image restoration quality 
and is at least partially related to the perceptual quality.
</pre>

**Command:**    
*Train.py:*
<pre>
train the SRCNN model using GPU, set learning rate=0.0005, batch size=256, 
make the program train 100 epoches and save a checkpoint every 10 epoches

python train.py train --cuda --lr=0.0005 --batch-size=256 --num-epoch=100 --save-freq=10
</pre>
<pre>
train the SRCNN model using CPU, set learning rate=0.001, batch size=128, 
make the program train 20 epoches and save a checkpoint every 2 epoches

python train.py train --lr=0.001 --batch-size=128 --num-epoch=20 --save-freq=2
</pre>

<pre>
resume training with GPU from "checkpoint.x" with saved hyperparameters

python train.py resume checkpoint.x --cuda
</pre>

<pre>
resume training from "checkpoint.x" and override some of saved hyperparameters

python train.py resume checkpoint.x --batch-size=16 --num-epoch=200
</pre>

<pre>
inspect "checkpoint.x"

python train.py inspect checkpoint.x
</pre>

*super_resolve.py*
<pre>
use the model stored in "checkpoint.x" to super resolve "lr.bmp"

python super_resolve.py --checkpoint checkpoint.x lr.bmp
</pre>