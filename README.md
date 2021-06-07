# msid-4
Metody Systemowe i Decyzyjne zadanie 4


#Fashion-MNIST ##Introduction The clue of the task is to implement a model that allows classification of thumbnails of photos representing clothes from Fashion-MNIST. Fashion-MNIST is a dataset of Zalando's article images-consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We'll be covering three classifiers: KNN and MLP.

Here we have an example of how the data looks:

![fashion-mnist](./image/fashion-mnist.png)


##First classifier we'll be covering is KNN (k-nearest neighbours). The Distance Metric I decided to use is Manhattan Distance. The KNN algorithm assumes that similar things exist in close proximity. K parameter represents number of neighbours we take into consideration.
![bestK](./image/bestK.png)



# How model looks like
![howModelLookLike](./image/howModelLookLike.png)

#Conv2D
Keras Conv2D is a 2D Convolution Layer, this layer creates a convolution kernel that is wind with layers input which helps produce a tensor of outputs.

Imagine we have an image that is an 2D array of 5x5 pixels. The idea of a Convolutional Layer is to essentially create another grid of layers known as the kernel or filter. So for example we have an 5x5 image and 3x3 kernel to filter the image. Convolution is an operation of multiplying corresponding kernel and pixels values, summing them up and passing to another array of pixels.

![keras_conv2d](./image/keras_conv2d.gif)


#MaxPooling2D
Downsamples the input along its spatial dimensions (height and width) by taking the maximum value over an input window (of size defined by pool_size) for each channel of the input. The window is shifted by strides along each dimension.

![MaxpoolSample2](./image/MaxpoolSample2.png)

![myAccu](./image/myAccu.png)

![accuPlot](./image/accuPlot.png)

