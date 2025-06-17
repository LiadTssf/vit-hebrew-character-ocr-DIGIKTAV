
dataset fore cnn hebrew letters - v1 2025-05-03 1:51pm
==============================

This dataset was exported via roboflow.com on May 3, 2025 at 10:59 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 92434 images.
Letters-WPAM are annotated in folder format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 72x72 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Randomly crop between 0 and 10 percent of the image
* Random rotation of between -3 and +3 degrees
* Random shear of between -6° to +6° horizontally and -6° to +6° vertically
* Random brigthness adjustment of between -13 and +13 percent
* Random exposure adjustment of between -10 and +10 percent
* Random Gaussian blur of between 0 and 1.1 pixels
* Salt and pepper noise was applied to 0.06 percent of pixels


