from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import mlflow
from ..speed_estimator.key_points import KeyPoints
from ..speed_estimator.neuflow import NeuFlow
from q_camera_top.testing.flow_vis import flow_to_image


class SpeedEstimatorTester(ABC):
    def __init__(self,
                 img_dir: Path):
        self.root = img_dir
        self.img1 = None
        self.img2 = None

    def get_imgs_id(self):
        ids = []
        for file in self.root.iterdir():
            id = file.stem.split('_')[-1]
            id = id.lstrip('0')
            ids.append(int(id))
        ids.sort()
        return ids

    def _find_paths(self, img_id: int):
        for file in self.root.iterdir():
            id1_pad = file.stem.split('_')[-1]
            id1 = id1_pad.lstrip('0')

            if int(id1) == img_id:
                img1_id = file

                numlen = len(id1_pad)
                id2 = str(int(id1) + 1)
                id2_pad = '0' * (numlen - len(id2)) + id2
                name = file.stem.split('_')
                name[-1] = id2_pad
                name = '_'.join(name)
                name += file.suffix
                img2_id = file.parent / name

                return img1_id, img2_id

    @abstractmethod
    def run():
        pass


class NeuFlowTester(SpeedEstimatorTester):
    def __init__(self,
                 img_dir: Path):
        super().__init__(img_dir)

        self.model: NeuFlow = NeuFlow()
        self.flow1 = None
        self.flow2 = None

    @staticmethod
    def _expand_flow(flow):
        """Принимает на вход u размерность потока (горизонтальный сдвиг) и расширяет его до (u, v) потока нулями."""
        return np.stack([flow, np.zeros_like(flow)], axis=2)

    def run(self, img_id):
        img1_path, img2_path = self._find_paths(img_id)

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        self.img1 = np.array(img1)
        self.img2 = np.array(img2)

        _ = self.model.get_momentum_speed(self.img1)

        _ = self.model.get_momentum_speed(self.img1)
        self.flow1 = self.model._prev_flow
        self.flow1 = self._expand_flow(self.flow1)

        speed = self.model.get_momentum_speed(self.img2)
        self.flow2 = self.model._prev_flow
        self.flow2 = self._expand_flow(self.flow2)

        return speed



class KPTester(SpeedEstimatorTester):
    def __init__(self,
                 img_dir: Path,
                 kp_type: str,
                 ):

        super().__init__(img_dir)
        self.kp: KeyPoints = KeyPoints(debug=True,
                                    #    roi=(0+350, 200, 1024-350, 768-200),
                                       kp_type=kp_type)
        self.kp_type = kp_type
        self.kp1 = None
        self.kp2 = None
        self.matches = None


    def vis_top_matches(self):
        matched_kp1_indices = [m.queryIdx for m in self.matches]
        matched_kp2_indices = [m.trainIdx for m in self.matches]

        new_matches = []
        for i, m in enumerate(self.matches):
            new_match = cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=m.distance)
            new_matches.append(new_match)

        kp1 = [self.kp1[i] for i in matched_kp1_indices]
        kp2 = [self.kp2[i] for i in matched_kp2_indices]

        img2plot = cv2.drawMatches(self.img1,
                                   kp1,
                                   self.img2,
                                   kp2,
                                   new_matches,
                                   None,
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return img2plot

    def vis_matches(self):
        img2plot = cv2.drawMatches(self.img1,
                                   self.kp1,
                                   self.img2,
                                   self.kp2,
                                   self.matches,
                                   None,
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return img2plot

    def vis_kp(self):
        img1 = cv2.drawKeypoints(self.img1, self.kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img2 = cv2.drawKeypoints(self.img2, self.kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img2plot = np.vstack([img1, img2])

        return img2plot

    def run(self, img_id: int):
        img1_path, img2_path = self._find_paths(img_id)

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        self.img1 = np.array(img1)
        self.img2 = np.array(img2)

        _ = self.kp.get_momentum_speed(self.img1)

        _ = self.kp.get_momentum_speed(self.img1)
        self.kp1 = self.kp.prev_keypoints

        speed = self.kp.get_momentum_speed(self.img2)
        self.kp2 = self.kp.prev_keypoints
        self.matches = self.kp.top_matches

        return speed


def log_image(img, key, step, scale=0.7):
    numlen = 4
    step = str(int(step) + 1)
    step_pad = '0' * (numlen - len(step)) + step
    img_scaled = cv2.resize(img, None, fx=scale, fy=scale)
    mlflow.log_image(img_scaled, key=key, step=step_pad)
    mlflow.log_image(img_scaled, artifact_file=key+'/'+str(step_pad)+'.png')


def log_kp(tester, i):
    kp = tester.vis_kp()
    log_image(kp, key=f'key_points', step=i)

    top_matches = tester.vis_top_matches()
    log_image(top_matches, key=f'top_matches', step=i)

    matches = tester.vis_matches()
    log_image(matches, key='all_matches', step=i)

def build_tester(kp_types, net_types, imgs_dir, estim_type):
    if estim_type in kp_types:
        tester = KPTester(imgs_dir, estim_type)
    elif estim_type in net_types:
        tester = NeuFlowTester(imgs_dir)
    else:
        raise ValueError('Incorrect speed estimator type.')
    return tester

if __name__ == '__main__':
    kp_types = ['orb', 'sift']
    net_types = ['neuflow']
    imgs_dir = Path('/home/devel/konyushenko/opticalFlow/framed_passes/3033_camera_top_far.mkv_20250305T093205.336800')
    # imgs_dir = Path('/home/devel/konyushenko/opticalFlow/framed_passes/1660_camera_top_far.mkv_20240424T131110.743152')
    estim_type = 'neuflow'

    imgs_dir_id = imgs_dir.stem.split('_')[0]
    mlflow.set_experiment(experiment_name=imgs_dir_id)
    for estim_type in net_types:# + kp_types:
        with mlflow.start_run(run_name=f'{imgs_dir_id}-{estim_type}-smROI'):
            tester = build_tester(kp_types, net_types, imgs_dir, estim_type)

            img_ids = tester.get_imgs_id()
            for i, id in enumerate(img_ids[:-1]):
                speed = tester.run(id)
                mlflow.log_metric('speed', np.abs(np.median(speed)), step=i)

                if estim_type in kp_types:
                    log_kp(tester, i)

                if estim_type in net_types:
                    log_image(flow_to_image(tester.flow2), key=f'flow', step=i)
                    # log_image(flow_to_image(tester.flow2 - tester.flow1), key=f'flow delta', step=i)
