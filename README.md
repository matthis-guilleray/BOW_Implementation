# Classification of Object :

## How to use it : 
Matlab is needed, to execute all the files
- The file "Detector.m" can be executed deep learning approachs
- The file "Classifier.m" can be executed to train the classifier and test the model Machine Learning approaches
- viewResult Take one parameter, the name of the result, defined in detector or result

## Things to do : 
- Implement hierarchical Kmean (bc 1H30min in Kmean)
- Function to store the Buck of word (only the words)
- Function to store the model trained
- Try the features detector with HOG, Fourier, dense SIFT descriptor
- Try different classifier
- Try parallel classifier
- Implement an optimizer of classifier
- Tweak the matching parameters (maybe decrease the percentage)
- Try different preprocessing of images
- Decreasing the number of keypoints
- Try to compare between a classic BOW of matlab
### This week-end : 
% Try to save the fd var, usig trycatch, plus run for SIFT, HOG, 
    % regarding this link : https://stackoverflow.com/questions/49963061/what-is-the-best-feature-detection
    % Try with : 
    % HOG
    % ORB (Evolution of SIFT and SURF)
    % AKAZE
    % BRISK (similar to ORB, but with high computational GPU)
    % Try to do it with very high number of VOC