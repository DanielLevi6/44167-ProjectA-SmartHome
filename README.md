# 44167-ProjectA-SmartHome
![OIG2 (1)](https://github.com/DanielLevi6/44167-ProjectA-SmartHome/assets/75536731/75263f1a-54f4-4579-b01e-3a3b969eac54)

## Introduction

Our project unveils an innovative smart home technology that operates through a hand-positioning control system.

Currently, IoT technology predominantly relies on voice commands, which can be challenging in loud environments or for individuals with speech difficulties. Hand-positioning control systems offer a solution to these challenges, providing an alternative means of interaction. Utilizing the Jetson Nano along with basic hardware components, we have developed a system that fulfills our requirements with notable accuracy and minimal response delay.

![Project (2)](https://github.com/DanielLevi6/44167-ProjectA-SmartHome/assets/75536731/37e6696d-ec25-4820-9447-9f853f143518)

## Goals
Develop a system based on an edge device that operates hardware solely through hand gestures.

This system should support six distinct functions. In our design, each of the first five functions will activate a corresponding LED, while the sixth function will switch off all LEDs.

The target is to achieve 90% accuracy on the test dataset

##  Developing tools
- Jetson Nano
- Keras
- MobileNet-V2
- OpenCV

## Model
Convolutional Neural Networks (CNNs) are the prevalent choice for image classification tasks due to their effectiveness in handling pixel data. In our quest for a CNN model that delivers high performance without consuming extensive system resources, we embarked on an in-depth exploration of the CNN landscape. Our research led us to identify two potential networks that appeared suitable for our requirements:

- Resnet18: Known for its deep residual learning framework, which facilitates the training of networks that are substantially deeper than those used previously.
- MobileNet-V2: Distinguished by its lightweight architecture that employs depth-wise separable convolutions to provide an efficient model size without compromising accuracy.

After thorough investigation and comparative analysis of both networks, we ultimately selected MobileNet-V2. Our decision was influenced by its compact structure and the efficiency it offers, making it an ideal solution for our system constraints while maintaining robust performance metrics.

## Dataset
For the sake of simplicity, we opted for a dataset featuring numerical sign language gestures. These gestures are easier to recall for individuals familiar with sign language, and they remain straightforward for those who are not.

![Picture1](https://github.com/DanielLevi6/44167-ProjectA-SmartHome/assets/75536731/884de17f-7fc9-473a-8802-89edaa3d3a7f)

Although the model was trained on all digits, we limited its use to the numbers 0 through 5. This strategy helped to minimize noise within the model. For instance, the digits 7 and 8 bear a close resemblance, as do 6 and 9. Such similarities can lead to confusion and increase the likelihood of the model’s failure.

(The dataset was taken from https://github.com/ardamavi/Sign-Language-Digits-Dataset)

## Training results
![acc graphs_after fine tuning](https://github.com/DanielLevi6/44167-ProjectA-SmartHome/assets/75536731/6861857a-8a9b-4457-bc57-ef4f3fea38b5)


## Program block diagram

![Hand Detector](https://github.com/DanielLevi6/44167-ProjectA-SmartHome/assets/75536731/5e277ebf-c24f-485f-8124-3d4922875164)
