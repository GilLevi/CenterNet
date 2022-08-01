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

# # Pose estimation on our data setting:
# TASK = 'multi_pose'
# MODEL_PATH = '/Users/gillevi/Projects/SurgeonAI/CenterNet/exp/multi_pose/default/model_10.pth'
# arch = 'res_18'


opt = opts().init('--task {} --load_model {} --arch {} --gpus -1 --debug 2'.format(TASK, MODEL_PATH, arch).split(' '))
# opt = opts().init('--task {} --load_model {} --arch {} --gpus -1 --debug 2 --dataset surgai'.format(TASK, MODEL_PATH, arch).split(' '))
# opt = opts().init(['--task=multi_pose', '--dataset=surgai', '--gpu=-1', '--arch=res_18', '--head_conv=64', '--num_workers=0', '--batch_size=1', '--debug=2','--load_model={}'.format(MODEL_PATH)])

# opt.heads = {'hm': 1, 'wh': 2, 'hps': 6, 'reg': 2, 'hm_hp': 3, 'hp_offset': 2}
Detector = detector_factory[opt.task]
detector = Detector(opt)

img_path = '/Users/gillevi/Downloads/yoav_wedding.jpeg'
# img_path = '/Users/gillevi/Projects/SurgeonAI/data/Hadassah_2021-07-05/raw_images/3/Frames/frame0067.png'
ret = detector.run(img_path)['results']
print(ret)