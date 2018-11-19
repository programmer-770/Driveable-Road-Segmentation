# Driveable-Road-Segmentation
This is an implementation for Driveable Road segmentation using Fully Convolution Network. The idea is inspired from the Paper "Semantic Segmentation using FCN" by Long et al.

For Downloading Pre-Trained VGG model-:https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip

For Downloading the KITTI Dataset-: http://www.cvlibs.net/download.php?file=data_road.zip

The pre-trained model(on KITTI Dataset) for Driveable Road Segmentation for direct inference can be found-: https://drive.google.com/drive/folders/16EmXscmBPXKgt5Nz35WaCNR_HHdFMNp_?usp=sharing

![alt text](https://meetshah1995.github.io/images/blog/ss/fcn.png)
A few key features of networks of this type are:

   -> The features are merged from different stages in the encoder which vary in coarseness of semantic information.
   -> The upsampling of learned low resolution semantic feature maps is done using deconvolutions which are initialized with            billinear interpolation filters.
   ->Excellent example for knowledge transfer from modern classifier networks like VGG16, Alexnet to perform semantic segmentation
![alt text](https://meetshah1995.github.io/images/blog/ss/fcn_1.png)
The fully connected layers (fc6, fc7) of classification networks like VGG16 were converted to fully convolutional layers and as shown in the figure above, it produces a class presence heatmap in low resolution, which then is upsampled using billinearly initialized deconvolutions and at each stage of upsampling further refined by fusing (simple addition) features from coarser but higher resolution feature maps from lower layers in the VGG 16 (conv4 and conv3) . A more detailed netscope-style visualization of the network can be found in at here

In conventional classification CNNs, pooling is used to increase the field of view and at the same time reduce the feature map resolution. While this works best for classification as the end goal is to just find the presence of a particular class, while the spatial location of the object is not of relevance. Thus pooling is introduced after each convolution block, to enable the succeeding block to extract more abstract, class-sailent features from the pooled features.

![alt text](https://meetshah1995.github.io/images/blog/ss/fcn_2.png)

On the other hand any sort of operation - pooling or strided convolutions is deterimental to for semantic segmentation as spatial information is lost. Most of the architectures listed below mainly differ in the mechanism employed by them in the decoder to recover the information lost while reducing the resolution in the encoder. As seen above, FCN-8s fused features from different coarseness (conv3, conv4 and fc7) to refine the segmentation using spatial information from different resolutions at different stages from the encoder.

The first conv layers captures low level geometric information and since this entrirely dataset dependent you notice the gradients adjusting the first layer weights to accustom the model to the dataset. Deeper conv layers from VGG have very small gradients flowing as the higher level semantic concepts captured here are good enough for segmentation. This is what amazes me about how well transfer learning works.

![alt text](https://meetshah1995.github.io/images/blog/ss/dilation.gif)

Other important aspect for a semantic segmentation architecture is the mechanism used for feature upsampling the low-resolution segmentation maps to input image resolution using learned deconvolutions or partially avoid the reduction of resolution altogether in the encoder using dilated convolutions at the cost of computation. Dilated convolutions are very expensive, even on modern GPUs. This post on distill.pub explains in a much more detail about deconvolutions.
