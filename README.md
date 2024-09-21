# Action-Recognition-on-YouTube-Shorts-Videos

This repository contains the implementation of an **LRCN model** (Long-term Recurrent Convolutional Network) for action recognition using the **UCF-101 dataset** and **YouTube videos**. The project applies a **key frame selection technique** based on optical flow and feature extraction to enhance model performance, especially on YouTube videos. This implementation is part of the research project titled "Automatic Audio Description Generation for YouTube Videos by Artificial Intelligence," supported by TUBITAK 2209-A.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Data Augmentation](#data-augmentation)
- [Key Frame Selection Method](#key-frame-selection-method)
- [Metrics and Results](#metrics-and-results)
- [Project Workflow](#project-workflow)
- [References](#references)

## Overview

This project focuses on recognizing actions in videos by classifying videos into one of eight categories selected from the UCF-101 dataset. While the model performs well on UCF-101, the performance on YouTube videos was initially poor. To address this, we developed a **key frame extraction** method that selects the most representative frames from each video using optical flow and ResNet50 feature extraction.

## Dataset

We used a subset of the UCF-101 dataset, focusing on the following 8 action classes:
- ApplyEyeMakeup
- BasketballDunk
- Diving
- IceDancing
- HorseRiding
- PlayingGuitar
- BlowingCandles
- WalkingWithDog


From each video, 5 frames were extracted.

## Data Augmentation

We applied the following augmentation techniques:
1. **Horizontal Flip**: Simulates left-right variation.
2. **Zooming (1.2x)**: Simulates different camera distances and angles.

 After data augmentation, the dataset sizes are:
- **Training Set**: 807 videos (2,421 augmented examples)
- **Test Set**: 333 videos (999 augmented examples)
- **YouTube Data**: 84 videos (each with extracted features and object vectors)

To ensure that variations of the same video do not appear in both the training and validation sets, we utilized the [UCF101 Train-Test Split Downloader](https://github.com/rocksyne/UCF101-train-test-split-downloader). This tool helps in downloading the dataset while maintaining the integrity of the train-test split.

## Model Architecture
The model consists of two parallel input streams:
1. **Frame Input**: Each frame is resized to 64x64 pixels and normalized.
2. **Objects Input**: Based on YOLOv3 object detection pre-trained with COCO dataset, object vectors of length 80 are generated for each frame.
   
The extracted features from the convolutional network are passed into an LSTM to capture the temporal information across video frames. The final output predicts the action class for each video.

<p align="center">
  <img src="https://github.com/user-attachments/assets/15df0990-074e-44fb-82b9-ec90891d8b9d" alt="LRCN Model Architecture" width="300">
</p>


## Key Frame Selection Method

To enhance performance on YouTube videos, we developed a key frame extraction method:

1. **Video Capture**: Extract individual frames from videos using OpenCV.
2. **Optical Flow Calculation**: Use Farneback optical flow to calculate motion magnitude between consecutive frames.
3. **ResNet50 Feature Extraction**: Extract semantic features from frames using a pre-trained ResNet50 model.
4. **Color Histogram Calculation**: Combine ResNet50 features and color histograms into a single feature vector.
5. **K-means Clustering**: Cluster frames based on feature vectors, selecting representative frames based on motion magnitude.

This approach ensures that selected frames capture significant actions and movements in the video.


## Metrics and Results

<table>
  <tr>
    <td>
      <h3>Key Frame Technique</h3>
      <table>
        <tr>
          <th>Metric</th>
          <th>Value</th>
        </tr>
        <tr>
          <td>Accuracy</td>
          <td>0.8810</td>
        </tr>
        <tr>
          <td>Precision</td>
          <td>0.8989</td>
        </tr>
        <tr>
          <td>Recall</td>
          <td>0.8810</td>
        </tr>
        <tr>
          <td>F1 Score</td>
          <td>0.8797</td>
        </tr>
      </table>
      <h4>Action Class</h4>
      <table>
        <tr>
          <th>Action Class</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>F1-Score</th>
          <th>Support</th>
        </tr>
        <tr>
          <td>ApplyEyeMakeup</td>
          <td>0.73</td>
          <td>0.80</td>
          <td>0.76</td>
          <td>10</td>
        </tr>
        <tr>
          <td>BasketballDunk</td>
          <td>1.00</td>
          <td>1.00</td>
          <td>1.00</td>
          <td>11</td>
        </tr>
        <tr>
          <td>Diving</td>
          <td>0.77</td>
          <td>1.00</td>
          <td>0.87</td>
          <td>10</td>
        </tr>
        <tr>
          <td>IceDancing</td>
          <td>0.90</td>
          <td>0.82</td>
          <td>0.86</td>
          <td>11</td>
        </tr>
        <tr>
          <td>HorseRiding</td>
          <td>0.79</td>
          <td>1.00</td>
          <td>0.88</td>
          <td>11</td>
        </tr>
        <tr>
          <td>PlayingGuitar</td>
          <td>1.00</td>
          <td>0.70</td>
          <td>0.82</td>
          <td>10</td>
        </tr>
        <tr>
          <td>BlowingCandles</td>
          <td>1.00</td>
          <td>1.00</td>
          <td>1.00</td>
          <td>11</td>
        </tr>
        <tr>
          <td>WalkingWithDog</td>
          <td>1.00</td>
          <td>0.70</td>
          <td>0.82</td>
          <td>10</td>
        </tr>
      </table>
    </td>
    <td>
      <h3>Uniform Sampling Technique</h3>
      <table>
        <tr>
          <th>Metric</th>
          <th>Value</th>
        </tr>
        <tr>
          <td>Accuracy</td>
          <td>0.7976</td>
        </tr>
        <tr>
          <td>Precision</td>
          <td>0.8280</td>
        </tr>
        <tr>
          <td>Recall</td>
          <td>0.7976</td>
        </tr>
        <tr>
          <td>F1 Score</td>
          <td>0.7902</td>
        </tr>
      </table>
      <h4>Action Class</h4>
      <table>
        <tr>
          <th>Action Class</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>F1-Score</th>
          <th>Support</th>
        </tr>
        <tr>
          <td>ApplyEyeMakeup</td>
          <td>0.90</td>
          <td>0.90</td>
          <td>0.90</td>
          <td>10</td>
        </tr>
        <tr>
          <td>BasketballDunk</td>
          <td>0.77</td>
          <td>0.91</td>
          <td>0.83</td>
          <td>11</td>
        </tr>
        <tr>
          <td>Diving</td>
          <td>0.67</td>
          <td>0.60</td>
          <td>0.63</td>
          <td>10</td>
        </tr>
        <tr>
          <td>IceDancing</td>
          <td>0.82</td>
          <td>0.82</td>
          <td>0.82</td>
          <td>11</td>
        </tr>
        <tr>
          <td>HorseRiding</td>
          <td>0.85</td>
          <td>1.00</td>
          <td>0.92</td>
          <td>11</td>
        </tr>
        <tr>
          <td>PlayingGuitar</td>
          <td>1.00</td>
          <td>0.50</td>
          <td>0.67</td>
          <td>10</td>
        </tr>
        <tr>
          <td>BlowingCandles</td>
          <td>0.65</td>
          <td>1.00</td>
          <td>0.79</td>
          <td>11</td>
        </tr>
        <tr>
          <td>WalkingWithDog</td>
          <td>1.00</td>
          <td>0.60</td>
          <td>0.75</td>
          <td>10</td>
        </tr>
      </table>
    </td>
  </tr>
</table>


| Key Frame Technique Confusion Matrix | Uniform Sampling Technique Confusion Matrix |
|--------------------------------------|---------------------------------------------|
| ![YouTubeKeyFrame_ConfusionMatrix_Accuracy_0.8809523582458496_Loss_1.0962570905685425](https://github.com/user-attachments/assets/35d4b7a7-ea51-4180-b4e7-e58ef8d7d0c6) | ![YouTube_ConfusionMatrix_Accuracy_0.7976190447807312_Loss_1.2761812210083008](https://github.com/user-attachments/assets/28255850-0dcf-45f3-9b64-ebbf20533bbb) |


## Project Workflow

1. **Configuration**:
   Modifying the `config` file optionally.

2. **Small Dataset (e.g. from YouTube) Creation**:
   A small dataset is created by collecting relevant videos from YouTube corresponding to the action classes of interest.

3. **Key Frame Extraction**:
   The script `extract_frames_better_class_level` is executed to extract key frames from the downloaded videos. This can be done either by downloading video-level data or directly at the dataset level.

4. **Train-Test Split**:
   Using the  [UCF101 Train-Test Split Downloader](https://github.com/rocksyne/UCF101-train-test-split-downloader)'s text files, containing paths of the relevant videos, train-test split scripts generate the necessary text files for the dataset.

5. **Dataset Creation**:
   The `create_dataset_` scripts are run to create datasets based on the generated text files or dataset origins.

6. **Object Detection**:
   The scripts in the yolo folder, specifically `class_and_count_`, are executed to extract object information from the relevant frames. This includes counting the number of detected objects and writing the results to text files.

7. **Conversion to PKL**:
   The `object_to_pkl` script is run to convert the data from the text files into PKL (Pickle) format for easier handling during model training.

8. **Model Training**:
   Finally, the `run` script is executed to train the model using the prepared dataset and features.
   

## References

1. **UCF-101 Dataset**: 
   K. Soomro, A. R. Zamir, and M. Shah. "UCF101: A Dataset of 101 Human Actions Classes From Videos in the Wild." *arXiv preprint arXiv:1212.0402* (2012). [Link to UCF101](https://www.crcv.ucf.edu/data/UCF101.php)
2. **YOLOv3 Model**:
   Joseph Redmon and Ali Farhadi. "YOLOv3: An Incremental Improvement." *arXiv preprint arXiv:1804.02767* (2018). [Link to YOLOv3](https://pjreddie.com/yolo/)
3. **UCF101 Train-Test Split Downloader**: 
   Rocksyne. "UCF101 Train-Test Split Downloader." [GitHub Repository](https://github.com/rocksyne/UCF101-train-test-split-downloader)
