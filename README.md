THIS IS A FORK THAT SUPPORTS Python3

python3 pyblur from [desmoteo](https://github.com/desmoteo/pyblur3)

# Running in Docker

## Build image
```
docker build -t syndata-generation .
```

## Run container
```
docker run -it syndata-generation bash
```

## Generate synthetic images (using demo data)
```
python3 dataset_generator.py ./demo_data_dir ./out --selected --add_distractors
```

## To reproduce results in the paper

0. get dataset.

I used BigBird dataset (contains 125 objects) for the object; UW Scene and GMU Kitchen dataset for the background.

1. train a FCN for the better foreground/background segmentation.

You can use the `FCN` folder for the reference. The FCN is trained on 125-11=114 objects and it would generate mask for the 11 objects we are going to detect.

2. once you have the mask, you can run `dataset_generator.py`

Note that you need to use the flag `--num 3` so that it can generate around 30000 synthetic images (as reported in the paper). But in the end I generated 39395 images. 

The command is 
```
python dataset_generator.py /path/to/bigbid /path/to/output/dir --add_distractors --selected --num 3
```

3. train detectron2!!

Once you have the synthetic images, you can train the object detection model!

I use detectron2 and you can use `train.py` as a reference. BTW you need to convert PASCAL VOC format as COCO format. You can use `convert_GMU_Kitchen_COCO.py` as a reference for conversion.

# Original README

## SynDataGeneration 

This code is used to generate synthetic scenes for the task of instance/object detection. Given images of objects in isolation from multiple views and some background scenes, it generates full scenes with multiple objects and annotations files which can be used to train an object detector. The approach used for generation works welll with region based object detection methods like [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn).

### Pre-requisites 
1. OpenCV (pip install opencv-python)
2. PIL (pip install Pillow)
3. Poisson Blending (Follow instructions [here](https://github.com/yskmt/pb)
4. PyBlur (pip install pyblur)

To be able to generate scenes this code assumes you have the object masks for all images. There is no pre-requisite on what algorithm is used to generate these masks as for different applications different algorithms might end up doing a good job. However, we recommend [Pixel Objectness with Bilinear Pooling](https://github.com/debidatta/pixelobjectness-bp) to automatically generate these masks. If you want to annotate the image manually we recommend GrabCut algorithms([here](https://github.com/opencv/opencv/blob/master/samples/python/grabcut.py), [here](https://github.com/cmuartfab/grabcut), [here](https://github.com/daviddoria/GrabCut))

### Setting up Defaults
The first section in the defaults.py file contains paths to various files and libraries. Set them up accordingly.

The other defaults refer to different image generating parameters that might be varied to produce scenes with different levels of clutter, occlusion, data augmentation etc. 

### Running the Script
```
python dataset_generator.py [-h] [--selected] [--scale] [--rotation]
                            [--num NUM] [--dontocclude] [--add_distractors]
                            root exp

Create dataset with different augmentations

positional arguments:
  root               The root directory which contains the images and
                     annotations.
  exp                The directory where images and annotation lists will be
                     created.

optional arguments:
  -h, --help         show this help message and exit
  --selected         Keep only selected instances in the test dataset. Default
                     is to keep all instances in the roo directory.
  --scale            Add scale augmentation.Default is to not add scale
                     augmentation.
  --rotation         Add rotation augmentation.Default is to not add rotation
                     augmentation.
  --num NUM          Number of times each image will be in dataset
  --dontocclude      Add objects without occlusion. Default is to produce
                     occlusions
  --add_distractors  Add distractors objects. Default is to not use
                     distractors
```

### Training an object detector
The code produces all the files required to train an object detector. The format is directly useful for Faster R-CNN but might be adapted for different object detectors too. The different files produced are:
1. __labels.txt__ - Contains the labels of the objects being trained
2. __annotations/*.xml__ - Contains annotation files in XML format which contain bounding box annotations for various scenes
3. __images/*.jpg__ - Contain image files of the synthetic scenes in JPEG format 
4. __train.txt__ - Contains list of synthetic image files and corresponding annotation files

There are tutorials describing how one can adapt Faster R-CNN code to run on a custom dataset like:
1. https://github.com/rbgirshick/py-faster-rcnn/issues/243
2. http://sgsai.blogspot.com/2016/02/training-faster-r-cnn-on-custom-dataset.html

### Paper

The code was used to generate synthetic scenes for the paper [Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection](https://arxiv.org/abs/1708.01642). 

If you find our code useful in your research, please consider citing:
```
@InProceedings{Dwibedi_2017_ICCV,
author = {Dwibedi, Debidatta and Misra, Ishan and Hebert, Martial},
title = {Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
```
