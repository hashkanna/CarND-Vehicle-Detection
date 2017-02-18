**Vehicle Detection Project**

[//]: # (Image References)

[image_0]: ./output_resources/vehicle_images.png "Vehicle Images"
[image_1]: ./output_resources/non_vehicle_images.png "Non Vehicle Images"
[image_2]: ./output_resources/sample_test_images.png "Test Images"
[image_2]: ./md_resources/image_2.png "Bit Mask Extraction"
[image_3]: ./md_resources/image_3.png "Histogram Point Fit"
[image_4]: ./md_resources/image_4.png "Polynomial Fit"
[image_5]: ./md_resources/image_5.png "Lane Augmented"
[image_6]: ./md_resources/image_6.png "Close Up"

The goals/steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier. - Done
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. - Done
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing. - Done
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images. - Done
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles. - Done
* Estimate a bounding box for vehicles detected. - Done


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

The code for this project can be found in the iPython notebook [Vehicle_Detection.ipynb](./Vehicle_Detection.ipynb)


The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
* Normalize the features and randomize a selection for training and testing
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   

###Histogram of Oriented Gradients (HOG)

####Criteria 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The images used are from GTI and KITTI datasets.
Python's glob package is used to read in all the images.
Two helper functions `load_images` and `plot_images` have been created to perform the loading and plotting of images. Refer [Vehicle_Detection.ipynb](./Vehicle_Detection.ipynb)

All `vehicle` and `non-vehicle` images are loaded using these functions.  
Here are some samples
Vehicle Images
![alt text][image_0]

Non Vehicle Images
![alt text][image_1]

Test Images
![alt text][image_2]
