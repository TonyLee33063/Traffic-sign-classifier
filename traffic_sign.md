# **Traffic Sign Recognition** 
---
### **Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the [data set](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)
* Explore, summarize,preprocess and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.
I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.
Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![bar chart](data/bar_chart.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the LeNet architecture needs 1 channel image inputs and we can just rely on the gray image to do the training with unnecessary redundant data. 
Here is an example of a traffic sign image before and after grayscaling.
![before](data/stright_or_right.png)
![afetr](data/stright_or_right2.png)
As a last step, I normalized the image data because it can prevents large data make big impacts and make the weights have nearlly the same effects.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5x32     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5x64 	    | 1x1 stride, valid padding, outputs 10x10x64 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				|
| Fully connected		| 320        									|
| dropout					|		0.75										|
| Fully connected		| 160        									|
| dropout					|		0.75										|
| output		| 43        									|
| Softmax				|         									|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, first I one-hot encoded the labels and using an learning rate of 0.001,batch size of 64 with 20 epochs and using the AdamOptimizer for graident descent to minimize the cross-entropy loss.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.948 
* test set accuracy of 0.926

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* At first I choose the LeNet-5 architecture for it has a very good performance on hand-written digit recognition and traffic-signs share some similarity,so I choose the LeNet-5 architecture.
* What were some problems with the initial architecture?
* But there are some shortcomings about the LeNet-5 architecture in this traffic-sign scenario for the image much more complex and have more patterns in it,as a result,the accuracy is not very high.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* So I adjust the model to have more filters at each convolutional layer such that it can detect much more patterns,and also to expand the size of the fully connected layers for it has much more classees. At the same time add some dropout layers to prevent overfitting and make the network more robust.
###Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![biycle crossing](data/Biyclej.jpg)
![double curve](data/double_curve.jpg)
![light](data/stopg.jpg)
![stop sign](data/60.jpg)
![right crossing](data/straightOrRight.jpg)

The first image might be difficult to classify because the bike doesn't have a good hierarchy structure. For the second image the curve might be the hard part and light image seems the circle number,and stop sign might be the semicycle part and the last one is the arrow part.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop Sign								| 
| 60(kmh) limit		| 60(kmh) limit										|
| Biycle crossing				| Biycle crossing									|
| Straight or right	      		| Straight or right					 				|
|Double curve		| Dangerous curve to the right      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares show a little lower than the test accuracy.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Stop sign   									| 
| .00016     				| Go straight or right 										|
| nearly 0					| Speed limit (30km/h)											|
| nearly 0	      			| Speed limit (50km/h)					 				|
| nearly 0				    | Go straight or left      							|


For the second image ... 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .997         			| Biycle crossing   									| 
| .00268     				| Slippery road 										|
| 0.0000539					| Beware of ice/snow											|
| nearly 0	      			| Turn left ahead					 				|
| nearly 0				    | Road work      							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Dangerous curve  to the right 									| 
| .0001    				| General caution										|
| nearly 0					| Children crossing											|
| nearly 0	      			| Speed limit (50km/h)Right-of-way at the next intersection					 				|
| nearly 0				    |Keep right     							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| straight or right 									| 
| nearly 0     				| Turn left ahead										|
| nearly 0					| Bumpy road											|
| nearly 0	      			| Ahead only					 				|
| nearly 0				    | No entry      							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9997         			| 60kmh limit   									| 
| .0002998     				| Speed limit (80km/h)									|
| nearly 0					| Wild animals crossing											|
| nearly 0	      			| Dangerous curve to the left					 				|
| nearly 0				    | Double curve     							|
