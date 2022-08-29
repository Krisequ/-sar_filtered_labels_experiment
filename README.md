# SAR Filter-Based Training

The following repository contains codes for two purposes:
- PNG image preprocessing
- Neural Network training

# Tif / Cos to png
The used databases consist of files with tif or cos extensions. The 3 python files enable conversion to more standard png file.

# PNG image preprocessing
The codes assume existance of "dataSrc" directory in the /src directory.
The dataSrc contains a database of SAR scans in png files.

If the images contain big parts of dark area (caused by casting of the imagery in the TIF to png conversion) 
trim_pngs.py contains utility for removing empty areas.

PNG files are processed in dataset_preparation.py, using images_preprocessing_lib. The goal of these scripts is
to split big (often bigger than 30'000 x 80'000 pixel images) into 1024x1024 images. The codes enable to set the overlap
between each image, the allowed empty space percentage in the image or the resolution. Additionally, the images are 
trimmed in different scales. The images, by default, are saved in "dataRdy/train".

With massive databases Overlap = 0 is preferred, to reduce the possibility of overtraining


After the small png are saved in "dataRdy/train" the train/val/test split should be performed using 
train_val_test_split.py. It does split the train/val and test big-scan-wise (so the AI cannot see the images with 
overlaps or different scales). The rest of the images are randomly split into val and train.

The default split is around 75%/20%/5%. The files names are saved to train_val_test_lists.py for ease of use. 

# Neural Network training

The training is performed using training.py file. For ease of multi-network training the training is performed using 
simple facade-like function performed for each of defined neural network configuration.

- model_definitions.py - keeps the architectures of the used neural networks
- utils.py - holds the callbacks / function loss definitions
- sequence_generator.py - declares the types of input/output image processing. This file also implements all the data 
  augmentation functions.


network_visualisation.py allows for performing benchmark of the trained neural networks, including 
- example image processing
- layer kernels extraction
- convolutional feature extraction-per layer

network_visualisation_lib.py contains the functions used by network_visualisation.py.


# Citation / deep dive

The algorithms used in here were described in the article [https://www.mdpi.com/1424-8220/22/17/6519](https://www.mdpi.com/1424-8220/22/17/6519)
For citation please use: 

AMA Style:
Gromada K. Unsupervised SAR Imagery Feature Learning with Median Filter-Based Loss Value. Sensors. 2022; 22(17):6519. https://doi.org/10.3390/s22176519

Chicago/Turabian Style:
Gromada, Krzysztof. 2022. "Unsupervised SAR Imagery Feature Learning with Median Filter-Based Loss Value" Sensors 22, no. 17: 6519. https://doi.org/10.3390/s22176519 


# Trained neural networks

The neural networks trained with these algorithms can be found under: 

[https://www.dropbox.com/s/izg31nd59ywf2h2/CNN_SAR_AE_Weigths.zip?dl=0](https://www.dropbox.com/s/izg31nd59ywf2h2/CNN_SAR_AE_Weigths.zip?dl=0)

The files containing weights, training logs and test images per-sub-epoch can be found:

[https://www.dropbox.com/s/t63ifout6gzs8kb/CNN_SAR_AE_Weigths_and_training.zip?dl=0](https://www.dropbox.com/s/t63ifout6gzs8kb/CNN_SAR_AE_Weigths_and_training.zip?dl=0)





