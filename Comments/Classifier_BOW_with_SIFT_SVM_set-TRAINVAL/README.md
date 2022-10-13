# Comments over the result of the use of the following algo : 
Implementation of Buck Of Word, using the following parameters : 
### Features detector : 
*SIFT* :
- Number of strongest points selected : 200
### Classifier : 
*SVM* :
- Number of image used for training : whole "trainval" set (2612 images)
### Vocabulary Building : 
- Number of words : 1500
- Number of images given in entry : The whole "trainval" set (2612 images)
    Kmeans algo implemented by Matlab
    - Number of iterations for the kmeans algo : 1000000
    - The distance for the kmean algo : 'sqeuclidean' 
    - Method to select the start points : 'Cluster'
### Feature matching : 
Algo implemented by Matlab
- Minimum percentage of matching : 5
### Testing :
Pre-constructed algo of the Pascal Project
- Number of images tested : The whole test set : 269

# Result achieved : 
It can be seen with the figure "SIFT + SVM Classifier.fig" file, 
Or the result file is the following one : result_file_comp1_bicycle.txt
Using the result file you can recompute the figure

But the Area under the curve it at 0.9
This result is not good bc the train set used included a part from the validation set

# Time and computational power
The computer used is an I7 8565u with an NVME ssd, and 16go of ram

Trace of the algorithm :

*00:14:50* - Starting of the algo
*00:14:50* - Starting the VOC Building
*00:14:50* - Reading of the images
*00:21:38* - Clustering algorithm
*Warning* : Failed to converge in 1000000 iterations in Kmeans functions
*01:39:27* - End of clustering algorithm - End of the VOC Building
*01:39:27* - Starting of the classifier Training
*01:39:27* - Reading of the images
*01:43:07* - Training of the model
*01:43:08* - End training of the model
*01:43:08* - Testing
Unknown  - End of testings
