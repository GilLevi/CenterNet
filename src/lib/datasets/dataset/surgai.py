from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pycocotools.coco as coco
import numpy as np
import json
import os
import torch.utils.data as data

class SurgAI(data.Dataset):
  num_classes = 1
  #default_resolution = [1920, 1080]
  default_resolution = [512, 512]
  # default_resolution = [1500, 1500]

  mean = np.array([0.36078363, 0.2696714 , 0.34761672],
                   dtype=np.float32).reshape(1, 1, 3)
  std = np.array([0.01912308, 0.01850544, 0.0194903],
                   dtype=np.float32).reshape(1, 1, 3)
  flip_idx = []
  num_joints = 3


  def __init__(self, opt, split):
    BASE_IMG_DIR = '/Users/gillevi/Projects/SurgeonAI/data/Hadassah_2021-07-05/raw_images_resized_1500'
    BASE_GT_DIR = '/Users/gillevi/Projects/SurgeonAI/data/Hadassah_2021-07-05/for_centernet/annotations_1500'
    super(SurgAI, self).__init__()
    if split == 'train':
        self.img_dir = os.path.join(BASE_IMG_DIR, '3/Frames')
        self.annot_path = os.path.join(BASE_GT_DIR, 'train_vid3.json')
    elif split == 'val':
        self.img_dir = os.path.join(BASE_IMG_DIR, '1/Frames')
        self.annot_path = os.path.join(BASE_GT_DIR, 'val_vid1.json')
    elif split == 'test':
        self.img_dir = os.path.join(BASE_IMG_DIR, '4/Frames')
        self.annot_path = os.path.join(BASE_GT_DIR, 'test_vid4.json')
    self.max_objs = 2
    self.class_name = ['__background__', 'P']
    self._valid_ids = [1, 2]
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]
    self._data_rng = np.random.RandomState(123)
    # used for color augmentations
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)

    self.split = split
    self.opt = opt

    #Gil: they use the variable name "coco' in kitti and pascal as well, I don't know why, but we'll use it here as well
    print('==> initializing coco 2017 {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  # I don't think this is needed, there are no calls for this function
  def convert_eval_format(self, all_bboxes):
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out  = list(map(self._to_float, bbox[0:4]))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(self._to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
                open('{}/results.json'.format(save_dir), 'w'))
  
  def run_eval(self, results, save_dir):
    self.save_results(results, save_dir)
