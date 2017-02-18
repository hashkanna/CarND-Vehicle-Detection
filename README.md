# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Goal
---
The goal of this project is to write a software pipeline to detect vehicles in a video

Vehicle Detection Project
---

[//]: # (Image References)

[image_0]: ./output_images/vehicles.png "Vehicle Images"
[image_1]: ./output_images/non_vehicles.png "Non Vehicle Images"
[image_2]: ./output_images/test_images.png "Test Images"
[image_3]: ./output_images/RGB_ColorTransform.png "Vehicle - RGB Color Transform"
[image_4]: ./output_images/HSV_ColorTransform.png "Vehicle - HSV Color Transform"
[image_5]: ./output_images/RGB_HOG.png "Vehicle - RGB HOG Transform"
[image_6]: ./output_images/HSV_HOG.png "Vehicle - HSV HOG Transform"
[image_7]: ./output_images/NonVehicle_RGB_ColorTransform.png "Non Vehicle - RGB Color Transform"
[image_8]: ./output_images/NonVehicle_HSV_ColorTransform.png "Non Vehicle - HSV Color Transform"
[image_9]: ./output_images/NonVehicle_RGB_HOG.png "Non Vehicle - RGB HOG Transform"
[image_10]: ./output_images/NonVehicle_HSV_HOG.png "Non Vehicle - HSV HOG Transform"
[image_11]: ./output_images/NonVehicle_HSV_HOG.png "Non Vehicle - HSV HOG Transform"


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

The sequence steps to perform the goal of this project is below:

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

#####Sample Vehicle Images
![alt text][image_0]

#####Sample Non-Vehicle Images
![alt text][image_1]

#####Sample Test Images
![alt text][image_2]

#####Color Spaces and HOG Features
Different color spaces like RGB, HSV have been explored.  

The hog package from scikit-Learn, skimage.hog(), has been used to extract the hog features

Different values for the various HOG parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) have been explored.

Here is an example using the RGB and HSV color spaces and HOG parameters as below

Param | Value
--- | ---
orientation | 8
pixels_per_cell | (8, 8)
cells_per_block | (2, 2)

#####Vehicle RGB Color Transform
![alt text][image_3]

#####Vehicle HSV Color Transform
![alt text][image_4]

#####Vehicle RGB HOG Transform
![alt text][image_5]

#####Vehicle HSV HOG Transform
![alt text][image_6]

#####Non-Vehicle RGB Color Transform
![alt text][image_7]

#####Non-Vehicle HSV Color Transform
![alt text][image_8]

#####Non-Vehicle RGB HOG Transform
![alt text][image_9]

#####Non-Vehicle HSV HOG Transform
![alt text][image_10]  

####Criteria 2. Explain how you settled on your final choice of HOG parameters

Various combinations of HOG parameters have been tried out manually and selected the ones which gave a good discrimination visually.
Parameter Grid Search could be done to automate this process and identify the best parameters.

####Criteria 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them)

Support Vector Classifier(LinearSVC) has been used to classify the vehicles and non-vehicles based on the training png images from KITTI [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images

* Training and Test Set has been created using the train_test_split function from sklearn. A ratio of 85:15 train/test split has been selected as shown in the code below
```
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
```

* The classifier pipeline has been chosen as below

```
clf = Pipeline([('feature_selection', SelectKBest(chi2, k=5000)),
                ('scaling', StandardScaler()),
                ('classification', LinearSVC(loss='hinge')),
               ])
```

* Model Evaluation has been performed using the score function from sklearn.
A score of 97.86% correct classification has been observed.
```
clf.score(X_test, y_test)
```

####Criteria 4. Sliding Window Search: Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

The code for the `slide_window` function can be found in [Vehicle_Detection.ipynb](./Vehicle_Detection.ipynb). Start/Stop parameters have been used to control the start and stop x and y positions of the window. window size and window overlap parameters have also been used.

Here is an example output of the `slide_window` function
![alt text][image_11]  
