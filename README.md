# Crime-Detection-and-Recognition-on-CCTV
To propose a new crime detection and recognition system using Computer Vision and Deep learning, which will definitely help to detect criminals from the emergence of CCTV.

**Dataset Description**

Crime recognitions from CCTV footage using UCF-crime dataset which can be obtained from kaggle. This version consists of 13 classes which includes abuse, arrest, arson, assault, burglary, explosion, fighting, shooting, stealing, shoplifting, road accidents and vandalism. These anomalies are selected because they have a significant impact on public safety.

**Pre-processing Data**

Input videos are converted into frames and take only 15 frames from every video for the training of models. These 15 frames were selected from a complete video sequence by skipping frames according to video length. Input frames are fed to the  Input layer of CNN and this layer pre-process the given images of any size and converts to 256*256*3 standard size.

**Feature Extraction**

The Convolutional Neural Network (CNN) used in this model, comprises an input convolutional layer   followed by three layers of convolution and max pooling. The kernel size for each convolutional layer is 2×2 . 32 kernels are used in each convolutional layer. The output from each convolutional layer after passing through “relu” activation function is max pooled to extract the features. Finally, the features are ﬂattened and sent to the next layer.

**Object Tracking**

It is done in video sequences CCTV surveillance feed and the objective is to track the path followed, speed of an object. The rate of real time detection can be increased by employing object tracking and running classification in a few frames captured in a fixed interval of time. 

**System Architecture**

The architecture has different phases like loading camera, video capture, image pre-processing, feature extraction, classification and prediction. The system classifies the videos into thirteen classes.



