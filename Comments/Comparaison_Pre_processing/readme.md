# Comparaison between different preprocessing filters
## Parameters used :
nbWordVOCBuilding = 200; % Vocabulary : K parameter in Kmeans in VOCBuilding
nbImagesVOCBuilding = 50; % Number of images processed in VOCBuilding, Max value at -1
nbImagesClsTraining = 50; % Number of images used to train the classifier, Max value at -1
nbThresholdMatchFct = 5; % Percentage needed to match between to features
nbImagesTesting = 100; % Number of images used to test the classifier, Max value at -1
maxIterationVOCBuilding = 1000000;
distanceAlgoVOCBuilding = 'sqeuclidean';
nbFdImages = 200; % Number of features descriptors per images
fdAlgo = "SIFT"; % 
trainingSet = "Train"

## Original :
With the only RGB to gray preprocessing of the Matlab function
Result O.15 over the bicycle test
Image n°1

## With a simple edge detector
Test realised while aplying a edge detector filter, using the log method after the RGB to gray processing (sigma value 2, size of the filter is n=ceil(sigma*3)*2+1)
Result of 0.156 over the bicycle test
Image n°2

## Gaussian + edge detector
We setup a gaussian filter of standard deviation of 0.5 and then we use the edge detector specified upper
Result of 0.100 over the bicycle test
Image n°3

## Canny edge detector :
Test realised using a canny edge filter sigma value is sqrt(2)
Result of 0.112 over the bicycle test
Image n°4

## Sobel filter
Test realised using sobel filter (default values)
Result of 0.126 over the bicycle test
Image n°5

## Log filter : 
Warning : The result are not stochastics
### SIGMA
Change in the SIGMA value : 
(Threshold set to a default value)
Default : 2 result : 0.156
SIMGA = 3 Result : 0.12
SIGMA = 1 Result : 0.17
SIMGA = 0.2 Result : 0.223
SIGMA = 0.01 Result : 0.168
SIGMA = 0.1 Result : 0.190
SIGMA = 0.15 Result : 0.196
SIGMA = 0.17 Result : 0.176
SIGMA = 0.19 Result : 0.153
SIGMA = 5 Result : Error

### THRESHOLD
Changes in the Threshold value :
(SIGMA set to 0.2)
Threshold = 1 Result : 0.129
Threshold = 2 Result : 0.157
Threshold = 3 Result : 0.149
Threshold = 0 Result : 0.16
Threshold = 20 Result : 0.157
Threshold = 30 Result : 0.169
Threshold = 40 Result : Error
Threshold = 35 Result : Error

