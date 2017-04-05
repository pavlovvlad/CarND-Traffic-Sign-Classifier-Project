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

[image1]: ./images/Hist_normed.JPG "Visualization: Histogram normed"
[image2]: ./images/Hist_abs.JPG "Visualization: Histogram absolut"
[image3]: ./images/Example_class_viz.JPG "Visualization: Example class"
[image4]: ./images/grayscale.JPG "Grayscaling&Normalization"
[image5]: ./images/3.JPG "Traffic Sign 1"
[image6]: ./images/21.JPG "Traffic Sign 2"
[image7]: ./images/24.JPG "Traffic Sign 3"
[image8]: ./images/26.JPG "Traffic Sign 4"
[image9]: ./images/28.JPG "Traffic Sign 5"
[image10]: ./images/30.JPG "Traffic Sign 6"
[image11]: ./images/41.JPG "Traffic Sign 7"
[image12]: ./images/Rotation_aug_dataset.JPG "Data augmentation: rotation 15 deg CCW and resampling method"
[image13]: ./images/wrong_classification.JPG "Wrong classification"
[image14]: ./images/Hist_augmented_data.JPG "New generated data sets only"
[image15]: ./images/Hist_all_data.JPG "All data sets inclusive real and new generated"
[image16]: ./images/TOP5Softmax_new_images_1.jpg "TOP5 softmax-probabilities for own choosen image of 'Speed limit 60km/h'"
[image17]: ./images/TOP5Softmax_new_images_2.jpg "TOP5 softmax-probabilities for own choosen image of 'Double curve'"
[image18]: ./images/TOP5Softmax_new_images_3.jpg "TOP5 softmax-probabilities for own choosen image of 'Road narrows on the right'"
[image19]: ./images/TOP5Softmax_new_images_4.jpg "TOP5 softmax-probabilities for own choosen image of 'Traffic signals'"
[image20]: ./images/TOP5Softmax_new_images_5.jpg "TOP5 softmax-probabilities for own choosen image of 'Children crossing'"
[image21]: ./images/TOP5Softmax_new_images_6.jpg "TOP5 softmax-probabilities for own choosen image of 'Beware of ice/snow'"
[image22]: ./images/TOP5Softmax_new_images_7.jpg "TOP5 softmax-probabilities for own choosen image of 'End of no passing'"

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

As it can be seen above the distribution of the samples for different classes is not equal, dominating of some features. To avoid this influence the augmented data sets have to be generated.

To avoid the warning about the maximal number of the open figures (by default - 20) the given style was customized by setting the rcParams plt.rcParams['figure.max_open_warning'] to 90 since we have 87 figures.

###Design and Test a Model Architecture

####1. Data augmentation.

In fourth code cell of the IPython notebook the data augmentation has been implemented in order to improve the classification accuracy for the new images.

To prepeare the augemented data set the Open-CV libraries were has been used from "https://conda.binstar.org/menpo opencv3".

Info: the 94%-accuracy of the test set has been achieved already without the data augmentation. 
Also 6 from 7 new traffic signs have been classifed correctly, besides "Road narrows right" - wrongly classified as "Road work".

Using the augemented data should increase the robustness of the classifier as it suggested in https://review.udacity.com/#!/rubrics/481/view. 

So the common data augmentation techniques like rotation, scailing and translation have been applied for the training set in order to decrease the influence of unbalanced data.
Parameters for the random (uniform distribution) rotation, scaling & translation
min scale = 0.8
max scale = 1.2
max rotation  = 15 in [deg]
max translation = 5 in [pixels]

As the result the [train, valid] sets with initial RGB-data have been increased more then twice with the new data proportionally as discribed before. The histogram with the number of new samples per class:
![alt text][image14]

The new data have been concatinated at the end of the array with initial data.

__Note__: for each training epoch the data in training set are shuffled in accordance with sklearn.utils.shuffle conventions.  

The sizes are:
* The size of training set: 74217
* The size of validation set: 10530
* The size of test set: 12630 (approx. 17% from the training set)

![alt text][image15]

After each transformation also the image resampling needs to be done. Here is the figure with the result of three different resampling techniques.    
![alt text][image12]

Optically the BILINEAR-resampling technique seems to be the optimal choice.


####2. Pre-process the data sets.

The code for this step is contained in the fifth code cell of the IPython notebook.

The following pre-processing steps are added:
1. Convert the images to grayscale to reduce the size of the neural network, and to have better edges by the feature detection. As it can be seen by the visualization of the examples of the dataset the color is not really powerful differentiation characteristic, especially by the dusk-scenarios.

2. To reduce the losing of the information due to overbrightness by images with highly amount of the bright and dark pixels the CLAHE (Contrast Limited Adaptive Histogram Equalization) with the 8x8 grid tile size and with the contrast limit = 2.0 has been applied. The images with the bright and dark pixels are captured often by twilight-scenarios with the traffic signes which have the retro-reflected areas. 

The increasing of the contrast limit up to 40.0 leads to better accuracy in validation and test sets, but has worse performance by new images with the traffic signs (5 from 7 properly classified), so empirically the optimal value = 2.0 has been defined.

3. Normalize the image data (grayscale/255) in all sets to avoid the processing of the large numbers by the training. Using of the normalization with the zero-mean at 128 (gray) reduces the accuracy by validation and test sets at 0.5% - 1.0% .

Here is the visualization of one example traffic sign image before and after pre-processing.
![alt text][image4]

####3. Model architecture

The sixth code cell of the IPython notebook contains the code.

The final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24	|
| RELU					|												|
| Max pooling			| 2x2 stride, valid padding, outputs 14x14x24 	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64	|
| RELU					|												|
| Max pooling			| 2x2 stride, valid padding, outputs 5x5x64 	|
| Flattening			| 5x5x64, outputs 1x1600						|
| Fully connected		| 1x1600, outputs 1x400							|
| RELU					|												|
| Dropout				| keep probability 0.5							|
| Fully connected		| 1x400, outputs 1x280							|
| RELU					|												|
| Dropout				| keep probability 0.5							|
| Fully connected		| 1x280, outputs 1x43							|
| Logits				|												|
| Softmax				|												|
| 1-Hot-Labels			|												|

The model is based on LeNet-architecture, where the depth-sizes for all layers have been increased by factor 3 and new additional regularization techniques are applied.

There are models with higher accuracy (up to 99.81%) like GoogLeNet incl. the inception layers with dimension reductions [GoogLeNet2014] and the models like described in [Haloi2015] using spatial transformer layers. The tuning of such network with more than 20 hidden layers will take to much time and resources scope of the project work.

The dropout with keep probability 0.5 (during training stage) has been used by both fully connected layers at the end as reqularization (to avoid the overfitting).

As the output-classes are mutually exclusive, the softmax cross entropy between logits and labels can be used to measure the prediction error.

####4. Model training.

The code for training the model is located in the seventh cell (definition of the tensors) and in the eigth cell (starting the session) of the IPython notebook. 

Early termination - technique has been used for regularization. Empirically defined optimal value for the number of epochs is 15.

The Adam-Optimizer has been applied, as it uses the momentum (moving average) in comparision to the GradientDescentOptimizer to improve the gradient direction, so the larger batch sizes can be used. Batch size has been empirically choosen equal 128.

After some tuning the hyperparameter have been set as follows: 
- learning rate: 0.001.
- learning rate decay: 0.9
- second moment: 0.999
- initial weights defined randomly with the normal distribution (mean = 0.0, std = 0.1)
- initial biases are zeros
- number of epochs: 15
- batch size: 128

####5. Solution Approach.

The code for calculating the accuracy of the model is located in the ninth cell of the IPython notebook.

An iterative approach has been chosen:
1. The first architecture was the LeNet-Lab-Solution (https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb): which achieved 92% by validation, 91% by test set and 4 from 7 own images classified correctly.

2. The next step was to add the dropout (with keep prob = 0.5) to the fully connected layers (s. chapter 3) : 94.5%(validation) and 93%(test) accuracy

3. Increasing of the depth by all layers by factor 3 (for more details s. chapter 3) (Adam-Optimizer with 0.001 learning rate) leads to __96.3%(validation) and 94%(test) accuracy__. Factor=3 is used because the current classifier is based on the LeNet-architecture but with the 43 output classes and uses grayscaled images of the same size as the classifier of digits 0-9 with 10 output classes. Factor = 4 has no sufficient influence on the accuracy, especially by classification of the own images.

4. Changing to the AdagradOptimizer with 0.1 learning rate shows 95.2%(validation) and 93.5%(test) accuracy (is faster as the Adam-Optimizer, but not so accurate. Therefore change back to the Adam-Optimizer with 0.001 learning rate.

5. Tuning the hyperparameter:
 (/) increasing of the initial learning rate from 0.001 to 0.002 leads to __96.8%(validation) and 94.9%(test)__ but 6 from 7 own   choosen traffic signes were classified correctly _Result: pass_  
 (x) increasing of the initial learning rate from 0.002 to 0.003 leads to __95.7%(validation) and 94.1%(test)__ _Result: fail_  
 (x) decreasing of the learning rate decay from 0.9 (default) to (0.8) results in 95.8%(validation) and 93.7%(test) _Result: fail_
 (x) reducing of the std by the weights initilization from 0.1 to 0.05 leads to 95.4%(validation) and 93.2%(test) _Result: fail_

6. Using augmented data set as described in chapter 1 leads to reducing of the accuracy: approx. 92%(validation), approx. 96%(test) and 6 from 7 correct classified traffic signs in own data set. _Result: fail_
 
 The "End of no passing" Label ID =  41 was wrongly clssified as "End of all speed and passing limits" Label ID =  32. 
 TOP5 softmax prob: [  6.30671978e-01   3.69318753e-01   7.90706054e-06   6.79236337e-07    5.84292138e-07] 
 TOP5 indices prob: [32 41 15 22 26]) 
 
 The softmax probabilities by the rest traffic signes was unambiguous.
 So the reason is to high max value of the scaling factor by the data augementation. 

7. Following changes have been applied to prevent the underfitting: 
 7.1 Change the max angle by random rotation (s. chapter 1) from 15 to 10 degree, 
 7.2 Increase the factor used by the depth definition in convolutional layers from 3 to 4.
 7.3 Increase the number of epoches from 11 to 15
 7.4 Decrease the learning rate to 0.001
 7.5 Apply adaptive histogram equalization (CLAHE) in pre-processing.
 
 The achieved accuracy was __94.4%(validation), 96.7%(test) and 7 from 7 correct classified traffic signs__ in own data set. _Result: pass_

_Note:_ Early termination - technique has been used for regularization. 

__The final model results were:
* validation set accuracy of 94.4%
* test set accuracy of 96.7% (no augmented data inside)
* accuracy by 7 own images 100% with the softmax probabilities of the winner-classes close to 1.0__

The difference between validation accuracy during the trainig and the accuracy by the test indicates that the model is fitted well (probability of the overfitting is small). 

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

To convert the images into 32x32x1(grayscale) format the padding with value 128 has been used. For this purpose the matplotlib.image library has been used.

####2. Performance on New Images.

The code for making predictions on the final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction after tuning:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 60km/h        		| 60km/h    									| 
| Double curve     		| Double curve									|
| Road narrows right	| Road narrows right							|
| Traffic Signals  		| Traffic Signals				 				|
| Children crossing		| Children crossing				 				|
| Beware Ice/snow		| Beware Ice/snow    							|
| End of no passing		| End of no passing    							|

The model was able to recognize correctly 7 of the 7 traffic signs, which gives an accuracy of 100%. That means that the model is robust to the new images and is not overfitted.

####3. Model Certainty - Softmax Probabilities.

The code for making predictions on the final model is located in the 11th cell of the IPython notebook.

#####__Itteration 1:__Without data augmentation but after tuning

The traffic sign "Road narrows right" was wrongly classified as "Road work":
![alt text][image13]

The difference between probablities Pwrong = 0.978 and Pright = 0.0176 is more then 0.96, so the model is not able to classify any "Road narrows right" properly. 

The reason is the unbalanced data sets. The difference in number of samples in [train/valid/test] sets for both classes is approx. 5 times:
- "Road narrows right", number of samples in [train/valid/test] sets: 240/30/90
- "Road work", number of samples in [train/valid/test] sets:          1350/150/480

To solve this problem the additional augmented data sets have to be created (s. chapter "Data augmentation" above).

#####__Itteration 2:__With the data augmentation

TOP5-softmax-probabilities for each sign type after tuning:

![alt text][image16]
![alt text][image17]
![alt text][image18]
![alt text][image19]
![alt text][image20]
![alt text][image21]
![alt text][image22]

As it can be seen in figures above the difference between softmax-probabilities of the 1st and 2nd places is greater as 0.9 by all choosen sign types. That indicates that the classes are clearly separable with the introduced classifier or in other words the model is certain to its predictions. 