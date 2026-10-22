import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import json
from .build import DATASETS
from utils.logger import *


@DATASETS.register_module()
class CrossCDR(data.Dataset):
    """
    CrossCDR: partial (clean / noisy) LiDAR scans of objects simulated in Unity, paired with
    complete point clouds sampled from the object meshes. Every sample folder contains
    *_normalized.ply partials and a full_point_cloud.ply, all in the normalized frame of the
    complete mesh (centered, fits inside the unit sphere).
    """
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.n_input_points = config.N_INPUT_POINTS
        self.partial_type = config.PARTIAL_TYPE
        self.subset = config.subset
        assert self.partial_type in ['clean', 'noisy'], f'Unknown PARTIAL_TYPE {self.partial_type}'

        self.split_file = os.path.join(self.data_root, f'{self.subset}_split.json')
        print_log(f'[DATASET] Open file {self.split_file}', logger='CrossCDR')
        with open(self.split_file, 'r') as f:
            records = json.load(f)['records']

        self.file_list = []
        for r in records:
            sample_dir = os.path.dirname(r['clean_point_cloud_path'])
            self.file_list.append({
                'taxonomy_id': r['object_type'],
                'model_id': sample_dir.replace('/', '_'),
                'partial_path': os.path.join(self.data_root, sample_dir, f'{self.partial_type}_normalized.ply'),
                'gt_path': os.path.join(self.data_root, sample_dir, 'full_point_cloud.ply'),
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded (partial type: {self.partial_type})', logger='CrossCDR')

        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'RandomMirrorPoints',
                'objects': ['partial', 'gt']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])

    @staticmethod
    def _resample(ptcloud, n_points, rng):
        """Randomly subsample to n_points, or keep all points and duplicate random ones (no zero padding)."""
        curr = ptcloud.shape[0]
        if curr >= n_points:
            choice = rng.choice(curr, n_points, replace=False)
        else:
            choice = np.concatenate([rng.permutation(curr), rng.choice(curr, n_points - curr, replace=True)])
        return ptcloud[choice]

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        # fixed per-sample seed outside of training so that val / test metrics are reproducible
        rng = np.random.default_rng() if self.subset == 'train' else np.random.default_rng(idx)

        data = {
            'partial': self._resample(IO.get(sample['partial_path']).astype(np.float32), self.n_input_points, rng),
            'gt': self._resample(IO.get(sample['gt_path']).astype(np.float32), self.npoints, rng),
        }

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

    def __len__(self):
        return len(self.file_list)
