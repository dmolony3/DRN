# Dilated Residual Network  
Tensorflow implementation of Dilated Residual Network from Fisher et al. 2017.  Contains
 both 18 layer and 26 layer implementations for **semantic segmentation**. Cross-entropy, weighted 
 cross-entropy and DICE loss can be used.
 
## Usage  
Users should enter values for *batch_size*, *num_epochs*, *image_dims* and *num_classes* in **train_network.py**

## Data Import  
Data is imported using the tensorflow Dataset API. A separate file for both train and validation/test
data should be generated. Each line in the file should contain the path to each image and it's label and weight

data_path/path_to_train_image data_path/path_to_label_image data_path/path_to_weight_image

![alt text](Example.png "Example image")
