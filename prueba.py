import os
import numpy as np
import random

import colorsys
import cv2

from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize

class CustomConfig(Config):
    NAME = "object"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 6  # background + objetos
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
    IMAGE_MAX_DIM = 1472
    DETECTION_MIN_CONFIDENCE = 0.9
inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", config=inference_config,  model_dir='models')
model_path = os.path.join('models', 'mask_rcnn_object_0009.h5')
model.load_weights(model_path, by_name=True)
print('ok')


camera = cv2.VideoCapture("video01.mp4")
class_names = ['BG', 'persona','p_comiendo','plato_vacio','plato_lleno','llajuero_lleno','llajuero_vacio']

import skimage.draw
img = skimage.io.imread('image06_624.png')
img_arr = np.array(img)
results = model.detect([img_arr], verbose=1)
print(results)
r = results[0]
print(r['class_ids'])
visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'], figsize=(5,5))
print('finish')
# while camera:
#     ret, frame = camera.read()
#     frame = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_AREA) 
#     results = model.detect([frame], verbose=0)
#     r = results[0]
    
#     N =  r['rois'].shape[0]
#     boxes=r['rois']
#     masks=r['masks']
#     class_ids=r['class_ids']
#     scores=r['scores']
    
       
#     hsv = [(i / N, 1, 0.7) for i in range(N)]
#     colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    
#     random.shuffle(colors)
#     #print("N_obj:",N)
#     masked_image = frame.astype(np.uint32).copy()
    
#     for i in range(N):
        
#         if not np.any(boxes[i]):
#             # Skip this instance. Has no bbox. Likely lost in image cropping.
#             continue

#         color = list(np.random.random(size=3) * 256)
#         mask = masks[:, :, i]
#         alpha=0.5

        
#         for c in range(3):
#             masked_image[:, :, c] = np.where(mask == 1,
#                                   masked_image[:, :, c] *
#                                   (1 - alpha) + alpha * color[c],
#                                   masked_image[:, :, c])
            
        
#         frame_obj=masked_image.astype(np.uint8)
#         y1, x1, y2, x2 = boxes[i]
#         cv2.rectangle(frame_obj, (x1, y1), (x2, y2),color, 2)  
        
#         class_id = class_ids[i]
#         score = scores[i] if scores is not None else None
#         label = class_names[class_id]
#         caption = "{} {:.3f}".format(label, score) if score else label
#         cv2.putText(frame_obj,caption,(int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#         masked_image = frame_obj.astype(np.uint32).copy()
    
        
#     if N>0:
#         cv2.imshow('frame', frame_obj)
#     else:
#         cv2.imshow('frame', frame)
    
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break
# print('break')     
# camera.release()
# cv2.destroyAllWindows()

# print('finish')