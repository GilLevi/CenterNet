from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pycocotools.coco as coco
import numpy as np
import json
import os
import torch.utils.data as data

gt_type = 'debug'
if gt_type == 'tiny':
  gt_dir = 'annotations_tiny'
elif gt_type == 'debug':
  gt_dir = 'annotations_debug_2' # annotations_debug_2
else:
  gt_dir = 'annotations'

class SurgAI(data.Dataset):
  num_classes = 1
  #default_resolution = [1920, 1080]
  default_resolution = [512, 512]

  mean = np.array([0.36078363, 0.2696714 , 0.34761672],
                   dtype=np.float32).reshape(1, 1, 3)
  std = np.array([0.01912308, 0.01850544, 0.0194903],
                   dtype=np.float32).reshape(1, 1, 3)
  # TODO: might need to set some value here
  flip_idx = []
  num_joints = 3 #TODO: hope it works

  def __init__(self, opt, split):
    super(SurgAI, self).__init__()
    # Gil: this is root data dir, I kept the original coco dir commented out as documentation
    self.data_dir = '/Users/gillevi/Projects/SurgeonAI/data/Hadassah_2021-07-05/for_centernet'
    # self.data_dir = os.path.join(opt.data_dir, 'coco')
    if split == 'train':
      self.img_dir = os.path.join(self.data_dir, 'train_vid3')
      self.img_dir = '/Users/gillevi/Projects/SurgeonAI/data/Hadassah_2021-07-05/raw_images_debug_resized/3/Frames'
      self.annot_path = os.path.join(self.data_dir, gt_dir, 'vid3.json')
    elif split == 'val':
      if gt_type == 'debug':
        self.img_dir = os.path.join(self.data_dir, 'train_vid3')
      else:
        self.img_dir = os.path.join(self.data_dir, 'val_vid4')
      self.annot_path = os.path.join(self.data_dir, gt_dir, 'vid3.json')
    elif split == 'test':
      if gt_type == 'debug':
        self.img_dir = os.path.join(self.data_dir, 'train_vid3')
      else:
        self.img_dir = os.path.join(self.data_dir, 'test_vid1')
      self.annot_path = os.path.join(self.data_dir, gt_dir, 'vid3.json')

    self.max_objs = 2
    #self.class_name = ['__background__', 'L', 'R']
    self.class_name = ['__background__', 'P']
    # TODO: I hope this corresponds to valid object categories
    self._valid_ids = [1, 2]
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]
    self._data_rng = np.random.RandomState(123)
    # TODO: I think this is used for color augmentations
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

  # TODO: I don't think this is needed
  def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()
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
  
  # TODO: this function is overloaded in each dataset and used to run evaluation, I will probably need to write my own
  def run_eval(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    #coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    #coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    #coco_eval.evaluate()
    #coco_eval.accumulate()
    #coco_eval.summarize()
    return None
