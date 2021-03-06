# -*- coding: utf-8 -*-
"""reconocimiento.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Q83MnizYXo3CabvaWbvey0PnX0bCcVS2
"""

!git clone https://github.com/vladip11/ProyectoSIS330.git

cd ProyectoSIS330/

!python setup.py install

import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw
import skimage.draw
import random
from mrcnn import visualize
from mrcnn.config import Config
from mrcnn import model as modellib, utils

ROOT_DIR = '/content/ProyectoSIS330/'
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist'
sys.path.append(ROOT_DIR)

class CustomConfig(Config):
    """Configuration for training on the helmet  dataset.
    """
    NAME = "object"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 4  # background + objetos
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1472
    STEPS_PER_EPOCH = 500
    VALIDATION_STEPS = 5
    BACKBONE = 'resnet50'
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50 
    POST_NMS_ROIS_INFERENCE = 500 
    POST_NMS_ROIS_TRAINING = 1000 
    
config = CustomConfig()
class InferenceConfig(CustomConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    #IMAGE_MIN_DIM = 512
    #IMAGE_MAX_DIM = 1472
    DETECTION_MIN_CONFIDENCE = 0.6
inference_config = InferenceConfig()

config.display()
inference_config.display()

model = modellib.MaskRCNN(mode="inference", config=inference_config,  model_dir='logs')

COCO_MODEL_PATH="/content/drive/MyDrive/miColab/mask_rcnn_object_0010.h5"
model.load_weights(COCO_MODEL_PATH, by_name=True)

class CustomDataset(utils.Dataset):
    def load_custom(self, dataset_dir, subset):
        self.add_class("object", 1, "persona")
        self.add_class("object", 2, "p_comiendo")
        self.add_class("object", 3, "plato_vacio")
        self.add_class("object", 4, "plato_lleno")
        assert subset in ["train", "val","test"]
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations1 = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations1.values())  
        annotations = [a for a in annotations if a['regions']]
 
        # Add images
        for a in annotations:          
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            objects = [s['region_attributes']['objects'] for s in a['regions'].values()]

            name_dict = {"persona": 1,"p_comiendo": 2,"plato_vacio":3,"plato_lleno":4}
            num_ids = [name_dict[a] for a in objects]
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
 
            self.add_image(
                "object",  
                image_id=a['filename'],  
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids)
 
    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids
 
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def IoU_function(mask_real, mask_results,plot_results=True):
    annotation_mask = np.zeros((mask_real.shape[0], mask.shape[1]))
    for i in range(mask_real.shape[2]):
        annotation_mask = np.logical_or(annotation_mask, mask_real[:,:,i])

    result_mask = np.zeros((mask_results.shape[0], mask_results.shape[1]))
    for i in range(r['masks'].shape[2]):
        result_mask = np.logical_or(result_mask, mask_results[:,:,i])
        #plt.imshow(result_mask, cmap='binary')
        #plt.show()
    
    if(plot_results):
        plt.title('Annotation Mask')
        plt.imshow(annotation_mask,cmap='binary')
        plt.show()
     
        plt.title('Result Mask')
        plt.imshow(result_mask, cmap='binary')
        plt.show()
        
    intersection = np.logical_and(annotation_mask, result_mask)
    union = np.logical_or(annotation_mask, result_mask)
    IoU = np.sum(intersection) / np.sum(union)
    
    if(plot_results):
        colors = visualize.random_colors(2)
        IoU_image = visualize.apply_mask(image, intersection,(1.0, 0.0, 0.0), alpha=0.8)
        IoU_image = visualize.apply_mask(IoU_image, union, (0.0, 1.0, 0.0), alpha=0.3)
        plt.figure(figsize = (10,10))
        plt.text(20, 30, 'Union',bbox=dict(facecolor='green', alpha=0.5))
        plt.text(20, 50, 'Intersection',bbox=dict(facecolor='red', alpha=0.5))
        plt.title('IoU Image - Intersection over Union')
        plt.imshow(IoU_image)
        plt.show()

    return IoU;

import skimage
import matplotlib.pyplot as plt
#prueba
real_test_dir = '/content/drive/MyDrive/miColab/DataSet/val'
dataset = CustomDataset()
dataset.load_custom("/content/drive/MyDrive/miColab/DataSet/", "val")
dataset.prepare()

IoU_list = []

for image_id in dataset.image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)

    results = model.detect([image], verbose=1)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                ['BG', 'persona', 'p_comiendo', 'plato_vacio', 'plato_lleno'], r['scores'], figsize=(5,5))

    IoU = IoU_function(mask, r['masks'])
    IoU_list.append(IoU)
    print('IoU:',IoU)
    print("classes detect: {},  classes real: {}  ".format(r['class_ids'],class_ids))

print("IoU's:{}".format(IoU_list))

print("classes detect: {},  classes real:  ".format(r['class_ids'],class_ids))

"""

image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        image_paths.append(os.path.join(real_test_dir, filename))

for image_path in image_paths:
    img = skimage.io.imread(image_path)
    if img.ndim != 3:
      image = skimage.color.gray2rgb(img)
    if img.shape[-1] == 4:
      image = img[..., :3]
    image = img[..., :3]
    img_arr = np.array(image)
    results = model.detect([img_arr], verbose=1)
    r = results[0]
    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                ['BG', 'persona', 'p_comiendo', 'plato_vacio', 'plato_lleno'], r['scores'], figsize=(5,5))
"""