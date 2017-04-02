#**Traffic Sign Recognition** 

##This writeup is based on the writeup template presented on https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/Hist_normed.jpg "Visualization: Histogram normed"
[image2]: ./images/Hist_abs.jpg "Visualization: Histogram absolut"
[image3]: ./images/Example_class_viz.jpg "Visualization: Example class"
[image4]: ./images/grayscale.jpg "Grayscaling&Normalization"
[image5]: ./images/3.jpg "Traffic Sign 1"
[image6]: ./images/21.jpg "Traffic Sign 2"
[image7]: ./images/24.jpg "Traffic Sign 3"
[image8]: ./images/26.jpg "Traffic Sign 4"
[image9]: ./images/28.jpg "Traffic Sign 5"
[image10]: ./images/30.jpg "Traffic Sign 6"
[image11]: ./images/41.jpg "Traffic Sign 7"
[image12]: ./images/Rotation_aug_dataset.jpg "Data augmentation: rotation 15 deg CCW and resampling method"
[image13]: ./images/wrong_classification.jpg "Wrong classification"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

Link to [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Literature:
[GoogLeNet2014](https://arxiv.org/abs/1409.4842) Szegedy,  C.,  Liu,  W.,  Jia,  Y.,  Sermanet,  P.,  Reed,  S.,  Anguelov,  D., Rabinovich,  A. "Going  deeper  with convolutions", 2014.
[Haloi2015](https://arxiv.org/pdf/1511.02992.pdf) M. Haloi, "Traffic Sign Classification Using Deep Inception Based Convolutional Networks", Guwahati, 2015.

###Data Set Summary & Exploration

####1. Summary of the data set. 

The code for this step is contained in the second code cell of the IPython notebook.  

* The size of training set: 34799
* The size of validation set:4410
* The size of test set: 12630 (approx. 36% from the training set)
* The shape of a traffic sign image (32x32x3) (RGB)
* The number of unique classes/labels in the data set is 43

To calculate summary statistics of the traffic signs data set the numpy library was used.

####2. Visualization of the dataset.

The code for this step is contained in the third code cell of the IPython notebook.  

The exploratory visualization consists in:
- two histograms showing the normalized and absolute distributions of the samples [train, valid, test] set over different classes
![alt text][image1]
![alt text][image2]
- a list of all traffic signs with a sample image, label-ID and amount of the samples [train, valid, test] set to each traffic sign 
![alt text][image3]

As it can be seen above the distribution of the samples for different classes is not equal, dominating of some features.

To avoid the warning about the maximal number of the open figures (by default - 20) the given style was customized by setting the rcParams plt.rcParams['figure.max_open_warning'] to 90 since we have 87 figures.

###Design and Test a Model Architecture

####1. Pre-process the data sets.

The code for this step is contained in the fourth code cell of the IPython notebook.

The following pre-processing steps are added:
1. Convert the images to grayscale to reduce the size of the neural network. As it can be seen by the visualization of the examples of the dataset the color is not really powerful differentiation characteristic, especially by the dusk-scenarios. Also by the grayscaled images the better contrast.

2. Normalize the image data and converting to the zero mean ((grayscale - 128)/128) in all sets to avoid the processing of the biased data and therefor for better condition (equal variance) by the training.

Here is the visualization of one example traffic sign image before and after grayscaling and also after normalizing.
![alt text][image4]

####2. Data augmentation.

In fifth code cell of the IPython notebook the data augmentation has been implemented in order to improve the classification accuracy for the new images.

Info: the 94%-accuracy of the test set has been achieved already without the data augmentation. 
Also 6 from 7 new traffic signs have been classifed correctly, besides "Road narrows right" - wrongly classified as "Road work".

Using the augemented data can increase the robustness of the classifier as it suggested in https://review.udacity.com/#!/rubrics/481/view. 

So the common data augmentation techniques like flip and rotation have been applied for the training set:
1. Flip: 
1.1 traffic signs of classes [11, 12, 13, 15, 17, 18, 22, 26, 30, 35] are symmetric and can be flipped vertically.
1.2 traffic signs of classes [19 <-> 20], [33 <-> 34], [36 <-> 37], [38 <-> 39] changing their class by the vertical flip. 

2. Rotation:
each sample of the classes not involved by the flip-transformation has been cloned and rotated with the random angle from a uniform distribution over [-15deg:15deg]. 

After rotation also the resampling needs to be done: here is the fingure with result of three different resampling techniques.    
![alt text][image12]

Optically the BILINEAR-resampling technique seems to be the optimal choise.

As the result the size of the training set has been increased with the augemented data by factor 2.

####3. Model architecture

The sixth code cell of the IPython notebook contains the code.

The final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x18	|
| RELU					|												|
| Average pooling		| 2x2 stride, valid padding, outputs 14x14x18 	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x48	|
| RELU					|												|
| Average pooling		| 2x2 stride, valid padding, outputs 5x5x48 	|
| Flattening			| 5x5x48, outputs 1x1200						|
| Fully connected		| 1x1200, outputs 1x400							|
| RELU					|												|
| Dropout				| keep probability 0.5							|
| Fully connected		| 1x1200, outputs 1x400							|
| RELU					|												|
| Dropout				| keep probability 0.5							|
| Fully connected		| 1x400, outputs 1x280							|
| Logits				|												|
| Softmax				|												|
| 1-Hot-Labels			|												|

The model is based on LeNet-architecture, where the depth-sizes for all layers have been increased by factor 3 and new additional regularization techniques are applied.

There are models with higher accuracy (up to 99.81%) like GoogLeNet incl. the inception layers with dimension reductions [GoogLeNet2014] and the models like described in [Haloi2015] using spatial transformer layers. The tuning of such network with more than 20 hidden layers will take to much time and resources scope of the project work.

The max-pooling after conv-layers have been replaced with the average-pooling to use the whole information from the input.

The dropout with keep probability 0.5 (during training stage) has been used by both fully connected layers at the end as reqularization (to avoid the overfitting).

As the output-classes are mutually exclusive, the softmax cross entropy between logits and labels can be used to measure the prediction error.

####4. Model training.

The code for training the model is located in the seventh cell (definition of the tensors) and in the eigth cell (starting the session) of the IPython notebook. 

Early termination - technique has been used for regularization. 
Empirically defined optimal value for the number of epochs is 11.

The Adam-Optimizer has been applied, as it uses the momentum (moving average) in comparision to the GradientDescentOptimizer to improve the gradient direction, so the larger batch sizes can be used. Batch size has been choosen equal 128, 11 epochs.

After some tuning the hyperparameter have been set as follows: 
- learning rate: 0.001.
- learning rate decay: 0.9
- second moment: 0.999
- initial weights defined randomly with the normal distribution (mean = 0, std = 0.1)
- initial biases are zeros
- number of epochs: 11
- batch size: 128

####5. Solution Approach.

The code for calculating the accuracy of the model is located in the ninth cell of the IPython notebook.

An iterative approach was chosen:
1. the first architecture was the LeNet-Lab-Solution (https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb): which achieved ?% by validation and ?% by test set.

2. next step was to add the dropout (with keep prob = 0.5) to the fully connected layers and average pooling to convolution layers: 94.5%(validation) and 93%(test) accuracy

3. increasing of the depth by all layers by factor 3 (Adam-Optimizer with 0.001 learning rate) leads to 96.3%(validation) and 94%(test) accuracy

4. changing to the AdagradOptimizer with 0.1 learning rate shows 95.2%(validation) and 93.5%(test) accuracy (is faster as the Adam-Optimizer, but not so accurate. Therefore change back to the Adam-Optimizer with 0.001 learning rate.

5. Tuning the hyperparameter:
- increasing of the learning rate from 0.001 to 0.005 leads to ...
- decreasing of the learning rate decay from 0.9 (default) to (0.8) results in ...
- increasing of the std by the weight initilization from 0.1 to 0.3 leads to ...


Note: Early termination - technique has been used for regularization. For now 11 epochs have been used to train the NN.

The final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?


The small difference between validation accuracy during the trainig and the accuracy by the test indicates that the model is fitted well (probability of the overfitting is small).

_If an iterative approach was chosen:
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
_
_If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


###Test a Model on New Images

####1. Acquiring New Images.

Here are seven German traffic signs found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10] ![alt text][image11]

- Image 1 "Speed limit 60km/h" might be difficult to classify because there are many similar classes of the traffic signs like 20, 30, 50, 70, 100 km/h. 
- Image 2 "Double curve" is similar to Image 3 "Road narrows to the right" and to Image "Road works" 
- Image 4 "Traffic Signals" is similar to Image "General caution"
- Image 5 "Children crossing" is similar to Image 3 "Road narrows to the right"
- Image 6 "Beware Ice/snow" can be confused with every traffic sign of the same form and with the figure inside.
- Image 7 "End of no passing" can be confound with Traffic Signs  "End of no passing by vehicles over 3.5 metric tons", "No passing", "no passing by vehicles over 3.5 metric tons"

To convert the images into 32x32x1(grayscale) format the padding with value 128 has been used. The value 128 represents a zero-mean by grayscaled image. For this purpose the PIL.Image library has been used.

####2. Performance on New Images.

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 60km/h        		| 60km/h    									| 
| Double curve     		| Double curve									|
| Road narrows right	| Road work										|
| Traffic Signals  		| Traffic Signals				 				|
| Children crossing		| Children crossing				 				|
| Beware Ice/snow		| Beware Ice/snow    							|
| End of no passing		| End of no passing    							|

The model was able to recognize correctly 6 of the 7 traffic signs, which gives an accuracy of 85.7%. This accuracy is close to the accuracy of the test set (94%). That means that the model robust to the new images and has high performance.

####3. Model Certainty - Softmax Probabilities.

The code for making predictions on the final model is located in the 11th cell of the IPython notebook.

The traffic sign "Road narrows right" was wrongly classified as "Road work":
![alt text][image13]

The difference between probablities Pwrong = 0.978 and Pright = 0.0176 is more then 0.96, so the model is not able to classify any "Road narrows right" properly. 

The reason is the unbalanced data sets. The difference in number of samples in [train/valid/test] sets for both classes is approx. 5 times:
- "Road narrows right", number of samples in [train/valid/test] sets: 240/30/90
- "Road work", number of samples in [train/valid/test] sets:          1350/150/480

To solve this problem the additional augmented data sets have to be created.

_Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 
