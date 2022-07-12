import sys
sys.path.insert(0, '/Users/gillevi/Projects/SurgeonAI/cocoapi/PythonAPI')
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

coco_annotation_file_path = "/Users/gillevi/Projects/SurgeonAI/data/Hadassah_2021-07-05/coco_test/train.json"

img_id = 2
im_path = '/Users/gillevi/Projects/SurgeonAI/data/Hadassah_2021-07-05/raw_images/1/Frames/frame1324.png'
coco_annotation = COCO(annotation_file=coco_annotation_file_path)
ann_ids = coco_annotation.getAnnIds(imgIds=[img_id], iscrowd=None)
anns = coco_annotation.loadAnns(ann_ids)
print(f"Annotations for Image ID {img_id}:")
im = Image.open(im_path)
plt.figure(figsize=(20,20))
plt.axis("off")
plt.imshow(np.asarray(im))
coco_annotation.showAnns(anns, draw_bbox=True)
plt.savefig(f"{img_id}_annotated.jpg", bbox_inches="tight", pad_inches=0)