# SageMaker Custom model development Hands-on Lab

The goal of this notebook is to show how to develop custom Machine Learning model on SageMaker for your business usecase. 
You will use the SageMaker built-in object detection algorithm and Tensorflow docker images to detect licene plate of a car and recognize the charcters of the plate by using CNN and computer vision technology. 

This HoL consists of four labs, each with the following details:

## [Lab1. Data preparation](Lab1-data-prep.ipynb)
- You will generate license plate images, cropped plate images, license plate position annotations, license plate characters annotations for training ML algorithm.  
- You will use gen-w-bbx.py with image synthesis  


<img src='imgs/Lab1.png' stype='width:600px;'/> . 
  
  
  
  
## [Lab2. Custom object detection with SageMaker built-in algorithm](Lab2-custom-object-detection.ipynb)
- SageMaker built-in algorithm
- Custom Object Detection 
- Transfer Learning (Resnet-50 base)  

<img src='imgs/Lab2.png' stype='width:600px;'/> . 
  
  
   
## [Lab3. Custom CNN script with Tensorflow and Keras](Lab3-custom-CNN-script-with-TF.ipynb)
- Custom Tensorflow script with Keras
- Custom CNN(Convolution Neural Net)  

<img src='imgs/Lab3.png' stype='width:600px;'/> . 
  
  
  
  
## [Lab4. Training, Deploying and hosting custom model on SageMaker](Lab4-train-deploy-host-on-SM.ipynb)
- Tensorflow script mode
- Distributed training
- Endpoint hosting  

<img src='imgs/Lab4.png' stype='width:600px;'/> . 
  
  
  

