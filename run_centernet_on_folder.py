import sys
sys.path.insert(0, '/Users/gillevi/Projects/SurgeonAI/CenterNet/src/lib')
sys.path.insert(0, '/Users/gillevi/Projects/SurgeonAI/CenterNet/src')
from opts import opts
from detectors.detector_factory import detector_factory
from pathlib import Path
from tqdm import tqdm
import json
# Pose estimation on our data setting:
ARCH = 'res_18'


def run_on_folder(detector, in_path: Path, out_path: Path):
    im_paths = sorted(list(in_path.glob('*.png')))
    for im_path in tqdm(im_paths):
        ret = detector.run(im_path.as_posix())
        with open(Path(out_path, im_path.name + '_res.json'), 'w+') as f:
            json.dump(ret, f)


def run_full_benchmark(epoch_list, videos_list, base_in_path: Path, base_out_path: Path):
    base_out_path.mkdir(parents=True, exist_ok=True)
    for epoch_num in epoch_list:
        cur_epoch_out_path = Path(base_out_path, f'epoch_{epoch_num:03d}')

        model_path = f'/Users/gillevi/Projects/SurgeonAI/CenterNet/exp/multi_pose/default_1500/model_{epoch_num}.pth'
        opt = opts().init(
            ['--task=multi_pose', '--dataset=surgai', '--gpu=-1', f'--arch={ARCH}', '--head_conv=-1', '--num_workers=0',
             '--batch_size=1', '--debug=0', '--load_model={}'.format(model_path)])

        Detector = detector_factory[opt.task]
        detector = Detector(opt)
        for vid_name in videos_list:
            print(f'processing epoch:{epoch_num}, vid:{vid_name}')
            cur_in_path = Path(base_in_path, vid_name, 'Frames')
            cur_out_path = Path(cur_epoch_out_path, vid_name)

            cur_out_path.mkdir(parents=True, exist_ok=True)
            run_on_folder(detector=detector, in_path=cur_in_path, out_path=cur_out_path)


if __name__ == '__main__':
    BASE_IN_PATH = '/Users/gillevi/Projects/SurgeonAI/data/Hadassah_2021-07-05/raw_images_resized_1500'
    BASE_OUT_PATH = '/Users/gillevi/Projects/SurgeonAI/data/results/Resnet18_1500_baseline'

    run_full_benchmark(epoch_list=[1, 5, 10, 15, 20, 25, 30], videos_list=['1', '3', '4'],
                       base_in_path=Path(BASE_IN_PATH), base_out_path=Path(BASE_OUT_PATH))
