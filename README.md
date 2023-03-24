# DeGPR
This is the official github repository of our paper DeGPR : Deep Guided Posterior Regularization For Multi-Class Cell Detection and Counting

## Abstract
Multi-class cell detection and counting is an essential task for many pathological diagnoses. Manual counting is tedious and often leads to inter-observer variations among pathologists. While there exist multiple, general-purpose, deep learning-based object detection and counting methods, they may not readily transfer to detecting and counting cells in medical images, due to the limited data, presence of tiny overlapping objects, multiple cell types, severe class-imbalance, minute differences in size/shape of cells, etc. In response,  we propose guided posterior regularization  DeGPR, which assists an object detector by guiding it to exploit discriminative features among cells. The features may be pathologist-provided or inferred directly from visual data.  
We validate our model on two publicly available datasets (CoNSeP and MoNuSAC), and on MuCeD, a novel dataset that we contribute. MuCeD consists of 55 biopsy images of the human duodenum for predicting celiac disease. We perform extensive experimentation with three object detection  baselines on three datasets to show that DeGPR is model-agnostic, and consistently improves baselines obtaining up to 9% (absolute) mAP gains. 

## Dataset Access
We are also releasing MuCeD dataset for multi-class cell detection and counting.
Please fill this form : https://forms.gle/fJ6kWt4RwXFgsxWh9 for access to MuCeD.

## Model Weights 
Weights from models can be downloaded using the link:https://drive.google.com/drive/folders/1rD6BeIaVdl27p4_x-4zdB8tmBl6jsaw1?usp=share_link
