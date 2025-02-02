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
# TASK = 'multi_pose'
# MODEL_PATH = '/Users/gillevi/Projects/SurgeonAI/CenterNet/models/multi_pose_hg_1x.pth'
# arch = 'hourglass'

# # Pose estimation on our data setting:
# TODO: 4CENTER_NET
TASK = 'multi_pose'
MODEL_PATH = '/Users/gillevi/Projects/SurgeonAI/CenterNet/exp/multi_pose/default_1500/model_30.pth'
# MODEL_PATH = '/Users/gillevi/Projects/SurgeonAI/CenterNet/exp/multi_pose/default_hourglass/model_4.pth'
arch = 'res_18'
# arch = 'hourglass'

# opt = opts().init('--task {} --load_model {} --arch {} --gpus -1 --debug 2'.format(TASK, MODEL_PATH, arch).split(' '))
# TODO: 4CENTER_NET
opt = opts().init(['--task=multi_pose', '--dataset=surgai', '--gpu=-1', f'--arch={arch}', '--head_conv=-1', '--num_workers=0', '--batch_size=1', '--debug=2','--load_model={}'.format(MODEL_PATH)])

# opt.heads = {'hm': 1, 'wh': 2, 'hps': 6, 'reg': 2, 'hm_hp': 3, 'hp_offset': 2}
Detector = detector_factory[opt.task]
detector = Detector(opt)

# img_path = '/Users/gillevi/Downloads/yoav_wedding.jpeg'
# img_path = '/Users/gillevi/Projects/SurgeonAI/data/Hadassah_2021-07-05/raw_images/3/Frames/frame0067.png'
img_path = '/Users/gillevi/Projects/SurgeonAI/data/Hadassah_2021-07-05/raw_images_resized_1500/3/Frames/frame7160.png'
ret = detector.run(img_path)['results']
print(ret)