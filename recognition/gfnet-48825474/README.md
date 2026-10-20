# Feasibility Review

The chosen model to implement is a GFNet on the ADNI dataset attempting to classify Alzheimer Disease (AD) vs Cognitively Normal (CN) conditions. 

1. User need, scope, and acceptance criteria
   1. The model will be used to predict and classify AD vs CN conditions based on input of brain MRI images. The case will be routed to a human for review if the model has low confidence in its classifcation.
   2. The prototype scope is to evaluate the accuracy of the GFNet model of this purpose by examining its performance, and computational trade-offs against other models for the same purpose.
   3. Testable criteria include: reduced computational cost compared to another model with similar performance measured in O(f(n)) and FLOPs (e.g. ViTs), 0.8 accuracy on AD vs CN, and max VRAM under 16GB
2. Model choice and course concepts (HARD difficulty)
   1. Proposed dataset configuration ADNI: dataset split into train and test (80:20) with each having CN and AD sub-folders
   2. GFNet architecture: Split image into patches, embed patches, global layer consisting of normalisation and 2D fast fourier transform (FFT), element-wise multiplicaiton between frequency domain features and learnable global filters, inverse FFT to convert back to spatial domain, and finally, a standard feed forward network.
   3. The original GFNet paper (Rao et al) found  the model to be feasible for classification on the CIFAR datasets.
   4. Course content on Transformer Neural Networks which is what GFNet's are heavily based off show that it is a suitable model for classification of images as it captures global contextual relationships between all features in the image.
3. Prelimary feasibility evidence
   1. ADNI data is adequate
   2. Rao et al. shows GFNet's efficiency and accuracy on RBG images which provides a reasonable proof that it works on images.
   3. Due to GFNet's O(L log L) time complexity, an EAIT lab computer with an RTX 2060 should be sufficient for smaller parameter configurations. Otherwise, upgrade to NVIDIa GPU sessions for more compute
4. Risk, Budget, and fallback
   1. Technical risks: patient data leakage between training samples, and leakage into test samples during split
   2. Compute budget: 27 hours of compute on NVIDIA data centre GPUs and queue times on Rangpur
   3. Fallback: 
   4. Next planned experiment: Apply GFNet to other medical image classification contexts (e.g. artery size impact on AD vs CN)

## Core Investigation

Metrics and benchmarks to consider: 

- Model architecture's ability to reliably quantify uncertainty on ambiguous boundary cases
  - Run inference and examine a test case where the outcome is ambiguous, and then compare the classification probability after using softmax.
- Comparison of performance to a simple baseline
  - Comparison of the Top-1 accuracy and Top-5 accuracy percentages.
  - Compare performance to a generic CNN and MLP baseline
  - Consider the number of parameters and FLOPs required for the different models
- Correlation of model's predicted confidence score with diagnotic correctness
  - Comparison of the diagnostic label of the true data versus what the model predicted.
  - Check for false negative and false positive cases, and measure the degree of incorrectness.

## Engineering Reccomendation



## Establish empirical confidence threshold or reject option rule that automates ambiguous cases to human specialists. 

- ## Whether the added complexity and computational cost justifies operational advantage over the baseline.