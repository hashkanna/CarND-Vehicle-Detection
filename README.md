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
[image_4]: ./output_images/HLS_ColorTransform.png "Vehicle - HLS Color Transform"
[image_5]: ./output_images/RGB_HOG.png "Vehicle - RGB HOG Transform"
[image_6]: ./output_images/HLS_HOG.png "Vehicle - HLS HOG Transform"
[image_7]: ./output_images/NonVehicle_RGB_ColorTransform.png "Non Vehicle - RGB Color Transform"
[image_8]: ./output_images/NonVehicle_HLS_ColorTransform.png "Non Vehicle - HLS Color Transform"
[image_9]: ./output_images/NonVehicle_RGB_HOG.png "Non Vehicle - RGB HOG Transform"
[image_10]: ./output_images/NonVehicle_HLS_HOG.png "Non Vehicle - HLS HOG Transform"
[image_11]: ./output_images/Slide_Window.png "Sliding Window"
[image_12]: ./output_images/Draw_Boxes.png "Draw Boxes"
[image_13]: ./output_images/Draw_Boxes1.png "Draw Boxes1"
[image_14]: ./output_images/Draw_Boxes2.png "Draw Boxes2"



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
Different color spaces like RGB, HSV, YUV, HLS have been explored.  Finally, YCrCb transform has been found to perform better. Details are in the notebook (Cells 5 and 127)

The hog package from scikit-Learn, skimage.hog(), has been used to extract the hog features

Different values for the various HOG parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) have been explored.

Here is an example using the RGB and HLS color spaces and HOG parameters as below

Param | Value
--- | ---
orientation | 9
pixels_per_cell | 12
cells_per_block | 2

#####Vehicle RGB Color Transform
![alt text][image_3]

#####Vehicle HLS Color Transform
![alt text][image_4]

#####Vehicle RGB HOG Transform
![alt text][image_5]

#####Vehicle HLS HOG Transform
![alt text][image_6]

#####Non-Vehicle RGB Color Transform
![alt text][image_7]

#####Non-Vehicle HLS Color Transform
![alt text][image_8]

#####Non-Vehicle RGB HOG Transform
![alt text][image_9]

#####Non-Vehicle HLS HOG Transform
![alt text][image_10]  

####Criteria 2. Explain how you settled on your final choice of HOG parameters

Various combinations of HOG parameters have been tried out manually and selected the ones which gave a good discrimination visually. Tried out 8, 12 and 16 pixels per cell and found 12 to perform better. There is a lot of turning of knobs of the various parameters with different values to get a better performance. Details are in Cell 5 of the notebook.

Parameter Grid Search could be done to automate this process and identify the best parameters.

The final choice of HOG parameters is shown below
Param | Value
--- | ---
orientation | 9
pixels_per_cell | 12
cells_per_block | 2

####Criteria 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them)

Support Vector Classifier(LinearSVC) has been used to classify the vehicles and non-vehicles based on the training png images from KITTI [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images

Code is available in Cells 129 and 130 of the notebook.

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

The code for the `slide_window` function can be found in Cell 131 of the notebook [Vehicle_Detection.ipynb](./Vehicle_Detection.ipynb). Start/Stop parameters have been used to control the start and stop x and y positions of the window. window size and window overlap parameters have also been used as mentioned in table below. The values 70, 120, 180 and 240 for the window sizes with a overlap of 80% turned out to be better empirically in this case.

Param | Value
--- | ---
window sizes | [(70, 70), (120,120), (180, 180), (240, 240)]
window overlaps | 0.8
Y Position | Range between 400 and 620
X Position | Entire range of the x values of the image from 0 to 1200


Here is an example output of the `slide_window` function
![alt text][image_11]  


####Criteria 5. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?
Here are some functions used to draw the bounding boxes, finding the cars, etc.
* `draw_boxes(img, bboxes, color=(0, 0, 255), thick=6)` - to draw boxes around the cars
* `find_cars(img, window_list, window_size, clf, train_height=64, train_width=64)` - To extract HOG features and predict if there is a car in the window.
* `find_cars_multiSize` - To find cars in different sizes of windows


Here are some images from the pipeline:
![alt text][image_12]  
![alt text][image_13]  
![alt text][image_14]  

---

### Video Implementation

####Criteria 6. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
[Result of Test Video](./test_video_out.mp4)

[Result of Project Video](./project_video_out.mp4)

####Criteria 7. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Postions of positive detections in each video frame has been recorded and a heatmap is created. Various thresholds have been tried to filter out and map the vehicle positions using `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. Bounding boxes are then created around each of the blob. Lots of false positives have been observed using this technique. One of the experiments that has been tried out to reduce the false positives is to exponentially decay the heatmap of prior filters. The functions to perform these (`add_heat`, `apply_threshold`, `bbox_from_labels`, `bboxes_from_detections`) are in the iPython notebook.

```
global heatmap_cont
heatmap = np.zeros((img.shape[0],img.shape[1]))
heatmap = add_heat(heatmap, detections)

heatmap_cont = heatmap_cont if not heatmap_cont is None else (heatmap*2)
heatmap += heatmap_cont
heatmap_cont = heatmap * 3/4

heatmap = apply_threshold(heatmap, threshold)
labels = label(heatmap)
```

Thresholds:  
A decision function has been used with Support Vector Classifier to let only windows that satisfy a  threshold greater than 0.95.
A threshold of 2 has been set for the heatmap.
Heatmaps of the bounding boxes of a rolling window of the most recent 40 frames are used to help with averaging and smoothing

---

###Discussion

####Criteria 8. Briefly, discuss any problems/issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here are some improvements that can be done to improve the project
* Use Deep Learning ConvNets to generate features automatically
* Parameter Selection - Use grid search or RandomizedGridSearch to scan the matrix of parameters to identify the optimal parameters instead of manually trying out arbitrary parameters
* False positives can be removed may be by adjusting the heat map or adding new features like Color Histograms
* Kalman filters could be a good addition
* The processing of the images have been slow. Optimizations can be performed by identifying the right algorithms and reduce feature selection or improve computing power
