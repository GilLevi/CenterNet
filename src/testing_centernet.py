# import _init_paths
import sys
sys.path.insert(0, '/Users/gillevi/Projects/SurgeonAI/CenterNet/src/lib')
sys.path.insert(0, '/Users/gillevi/Projects/SurgeonAI/CenterNet/src')
from opts import opts
from detectors.detector_factory import detector_factory

# Detection setting
# TASK = 'ctdet'
# MODEL_PATH = '/Users/gillevi/Projects/SurgeonAI/CenterNet/models/ctdet_coco_hg.pth'
# arch = 'hourglass'

# Pose estimation setting
TASK = 'multi_pose'
MODEL_PATH = '/Users/gillevi/Projects/SurgeonAI/CenterNet/models/multi_pose_hg_1x.pth'
arch = 'hourglass'

# Pose estimation on our data setting:
TASK = 'multi_pose'
MODEL_PATH = '/Users/gillevi/Projects/SurgeonAI/CenterNet/exp/multi_pose/default/model_last.pth'
arch = 'res_18'



#opt = opts().init('--task {} --load_model {} --arch {} --gpus -1 --debug 2'.format(TASK, MODEL_PATH, arch).split(' '))
opt = opts().init('--task {} --load_model {} --arch {} --gpus -1 --debug 2 --dataset surgai'.format(TASK, MODEL_PATH, arch).split(' '))
Detector = detector_factory[opt.task]
detector = Detector(opt)

# img_path = '/Users/gillevi/Downloads/yoav_wedding.jpeg'
img_path = '/Users/gillevi/Projects/SurgeonAI/data/Hadassah_2021-07-05/raw_images/1/Frames/frame1971.png'
ret = detector.run(img_path)['results']
print(ret)