# Dilated Residual Network  
Tensorflow implementation of Dilated Residual Network from Fisher et al. 2017.  Contains
 both 18 layer and 26 layer implementations for **semantic segmentation**. Cross-entropy, weighted 
 cross-entropy and DICE loss can be used.
 
## Usage  
The model should be trained from the command line. Users should enter values for *image_dims*, *num_classes*, *channels*, *train_file* and 
*val_file*. The entry for train_file and path_file should be paths to text files where each line is a data sample consisting of the image path, 
the label path and a label weight path. The image file should be jpeg while the label and weight files should be png. These are separated by spaces as below. Alternatively, either or both the label and weight file can be ommitted.

```
path_to_train_image1.jpg path_to_label_image1.png path_to_weight_image1.png
path_to_train_image2.jpg path_to_label_image2.png path_to_weight_image2.png
```

An example of an image, labelmap and weightmap is shown below.
![Example](Example.png "Example image")

The model can be trained with the following arguments

```
python main.py --train_file=path/to/train_data.txt --val_file=path/to/val_data --directory=image/directory_path --num_classes=5 --image_dims=500 --channels=3 --batch_size=8 --loss=CE
```

The model can be used for inference by setting the mode to predict and providing similar arguments as above
```
python main.py --mode=predict --val_file=path_to_val_data.txt --directory=C:\Users\David\Desktop\ivus_images1 --num_classes=5 --image_dims=500 --channels=3
```

