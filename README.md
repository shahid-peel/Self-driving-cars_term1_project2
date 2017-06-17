# ** Traffic Sign Recognition ** 

## Project Writeup

---

** Traffic Sign Recognition Project **

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

'', '', '', '', ''

[//]: # (Image References)

[image1]: ./examples/random_sign.png "Randon Sign visualization"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_images/speed_30.jpg "30 kph speed limit"
[image5]: ./test_images/speed_60.jpg "60 kph speed limit"
[image6]: ./test_images/stay_right.jpg "Stay right"
[image7]: ./test_images/stop.jpg "No entry"
[image8]: ./test_images/work.jpg "Road work"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it!

### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43


Here is an exploratory visualization of the data set. Just to make sure that the image data was loaded correctly from the pickle, I randomly picked an image and displayed it.  With this particular dataset I didn't visualize more as I understand that the dataset has been vetted by many project submission earlier.  if this was a one off project/work i was doing, I would visualize the different classes, here did the minimum i needed to do a sanity check.

![alt text][image1]

### Design and Test a Model Architecture

Preprocessed the data by normalizing it.  Ensured that the values of rgb were between -1.0 and 1.0.  Also when my initial model didn't give results better than 90%, i came back to this step and experimented with converting images to grayscale as a pre-processing step.  That gave some problems with dataset dimensionality so I ended us doing the conversion through a tensorflow api.  With that the results were better.


For the model, I based it on the tried and tested LeNet, with some modifications of course.  Modification were initially in both the number of channels of the input images and the number of final outputs (classes).  Later on as I started grayscaling the input images, the number of channels went back to that of LeNet i.e. single grayscale channel.

Layers of the final model:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16  	|
| RELU					|									            | 
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten		        | outputs [400]        						    | 	
| Fully connected		| input = 400, output = 120        				| 
| RELU					|									     		|
| Fully connected		| input = 120, output = 84        				| 
| RELU					|									            | 
| Fully connected		| input = 84, output = 43        				| 
| Softmax				| etc.        									|
|						|												|


#### Model Training

Initially I was getting an accuracy on the validation set of around 88%.  That was with just mimicking my network and parameters based on LeNet.  I played around with 3 parameters:
- learning rate
- batch size
- epocs

I increased the nubmer of epocs as I could see that the model was still improving as the number of epocs was going up.  Next I played around with the learning rate.  Increasing the learning rate too much caused the training to diverge and not converge on a high accuracy.  Decreasing it slowed down the learning process too much.  The batch size was something that I wanted to decrease to some extent so as to give more chances for the model to udpate the weights.  In the end the final values of the 3 hyperparameters were:
Epocs = 30
Batch size = 64
Learning rate = 0.003

#### Approach Taken

My final model results were:
* validation set accuracy of 94.9
* test set accuracy of 91.3

I chose a well known architecture:
* Architecture chosen: LeNet
* I selected this architecture mainly because that is the only working one that was available to me and so it provided a reference point/platform to get started with.  Also the content of the images seems similar e.g. speed values are numbers (similar to alphabets) and there are lots of circular and triangular shapes, ones that are there in the case of handwriting/character recognition.  
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


### Testing Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first two image were the speed limit signs.  I chose one with the blue (sky) background whereas the other was with the terrain background.  Also there was some external noise in the form of image watermarks in there as well.  That added an additional level of difficulty to the test - perhaps it can be raining or the camera sensor might have spots on lens, so these type of tests are good.

I also had images of high resolution as I wanted to see how a significant scaling down of image might effect the classification.  Example of this image was the 'stay right' sign and the 'road work' sign.

Lastly the 'no entry' sign was not centered in the image and was to the side. I wanted to see how the model would classify this image.   

So all in all these were a mix of different test situations we might expect to see in real world.


#### Model's predictions on these new traffic signs

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 60 kph sign      		| 60 kph sign   								| 
| 30 kph sign  			| 30 kph sign 									|
| Stay right			| Stay right									|
| No entry	      		| No entry					 					|
| Road work 			| Road work      								|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 91.3%.  This means that despite the fact that there was noise in the form of image watermarks, the output was not effected, infact we did better than what we expected.  Ofcourse 5 images is a very small number and the 100% isn't an accurate reflection of what we would encounter 'in the wild'.

#### Model certainty

The model was very certain when predicting most of the classes.  For 3 of them it got very close to 100% probability.  With one of them it was 99% and the lowest overall was 76%.

For the first image, the model is least sure as compared to the other signs.  The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .76        			| 30 kph   										| 
| .23     				| General caution 								|

This was the worst result of all, because the 2nd most likely class choice isn't even close to the round speed limit sign.  

For the second image there was a lot of difference between the 1st and 2nd choice (enen though the 2nd choice is a similar sign).  This makes me think that the model might be overfitting a bit - but this hypothesis isn't verified by the overall test accuracy of 91.3%. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99        			| 60 kph   										| 
| .01     				| 50 kph 								|

For the 3rd, 4th, and 5th test images, the predication confidence was 100% in favor of the correct class.

#### Conclusion

Overall a fun project to learn CNN on.  The LeNet baseline helped save sometime and so allowed me to focus on the other aspects of the learning exercise.  



