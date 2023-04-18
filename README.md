# Computational-Data-Science
Group project for 50.038 Computational Data Science, SUTD Term 6
<br>
## Problem Statement

The development of deep fake technology has opened up new possibilities in the world of digital media, but it has also raised concerns about the authenticity of the content. The ability to manipulate images, videos, and audio with ease has made it increasingly difficult to distinguish between real and fake content. This has serious implications for the spread of fake news and misinformation, which can have far-reaching consequences in areas such as politics, healthcare, and finance.
Hence, this project aims to develop a robust and efficient deep fake detection model that can accurately identify manipulated media using various machine learning techniques, thereby helping prevent the spread of fake news and misinformation.

## Dataset and Collection

Our dataset consists of a huge collection of .mp4 video files (~470GB) of real and deepfake videos that have been split into compressed sets of approximately 10GB each. Each set of .mp4 files is accompanied by a metadata.json file that contains relevant information about the video files, including the filename, label (either "REAL" or "FAKE"), original, and split columns. The models have been trained on these dataset of videos and corresponding labels, with the goal of predicting the probability that a given video is a deepfake, regardless of its label.

## Data Pre-processing

### 1. Frame Extraction
As part of our data pre-processing process, we created a custom method called save_frame to extract individual frames from each of the real and fake videos. This was achieved by using OpenCV's VideoCapture function to open the video files and then read each frame from the video file and save it as an individual image file in a new directory.
This approach allowed us to easily break down a video file into its constituent frames for further processing or analysis. By separating the video into frames, we were able to analyse individual images and detect any patterns or anomalies in the data that were not visible in the video as a whole.

### 2. Face Detection & Image Cropping 

Additionally, we implemented the process_face method to extract and crop the face regions of individual images from our dataset. This is accomplished through the use of the face_recognition library and PIL's Image module. By locating the faces in an image, the method  crops out the images to just the face region, 
and saves these cropped images as new files. This is particularly significant in the context of deepfake classification as deepfakes usually heavily rely on the manipulation of facial features. By isolating just the face region, we can hence ensure that the models are focusing on the relevant features for classification


### 3. Image Sampling through randomization

Further on, since the process_face method generates a large volume of images from each video (approximately 150 frames per video), we chose a smaller number of frames to use in the training and testing to reduce the computational burden on our machine learning models. This was accomplished by randomly selecting 30 frames from each video which provided a good balance between computational efficiency and providing enough data for the models to learn from. This step is crucial in creating efficient and effective deepfake classification models as it reduces the amount of data that needs to be processed by the models while still providing enough data for the models to learn from. Additionally, selecting a random sample of frames from each video helps to avoid any bias that may be present in the original video and ensures that the models are exposed to a diverse set of images. 

### 4. Image Augmentation
Finally, we implemented image augmentation on our real and fake datasets by applying a variety of transformations to the original images, such as rotation, scaling, and flipping. This is useful as it helps to 
prevent overfitting, which is a common problem in deep learning models. By creating new images that are slightly different from the original ones, our models are forced to learn more robust and generalizable features that can be applied to a wider range of images. Moreover, image augmentation helped to balance the distribution of classes in our datasets, especially when dealing with imbalanced datasets, by generating more samples for the minority class (real data in our case).






