"""冻结的 ResNet18 特征与加性多核。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from rgbd_data import preprocess, save_json


def fingerprint(manifest):
    specification = {'manifest': manifest, 'preprocess': '224-bilinear-rgb-nearest-depth-train-q01q99-imagenet-v1',
                     'encoder': 'resnet18-imagenet1k-v1-l2-512'}
    return hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # 小分类器使用 CPU，限制线程避免小矩阵的线程调度开销。
    torch.set_num_threads(min(4, torch.get_num_threads()))


def choose_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Run: python main.py doctor')
    return torch.device(name)


class PairDataset(Dataset):
    def __init__(self, manifest, rows=None, occlusion=0.):
        self.manifest = manifest
        self.rows = manifest['rows'] if rows is None else rows
        self.occlusion = occlusion
        self.mean = torch.tensor([.485, .456, .406])[:, None, None]
        self.std = torch.tensor([.229, .224, .225])[:, None, None]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        rgb, depth = preprocess(self.rows[index], self.manifest['root'], self.manifest['depth'],
                                occlusion=self.occlusion, seed=self.manifest['seed'])
        def tensor(arr):
            image = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
            return (image - self.mean) / self.std
        return tensor(rgb), tensor(depth)


def make_encoder(device, pretrained=True):
    # ImageNet 权重缓存位于项目 cache/torch 下，由 main.py 设置。
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Identity()
    model.requires_grad_(False)
    return model.eval().to(device)


@torch.inference_mode()
def encode(manifest, encoder, device, batch_size=16, rows=None, occlusion=0., rgb_only=False):
    if batch_size < 1:
        raise ValueError('batch_size must be positive.')
    dataset = PairDataset(manifest, rows=rows, occlusion=occlusion)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == 'cuda')
    rgb_out, depth_out = [], []
    for i, (rgb, dep) in enumerate(loader):
        # 两种模态依次通过冻结的同一特征网络，降低显存占用。
        rgb_out.append(F.normalize(encoder(rgb.to(device)), dim=1).cpu().numpy())
        if not rgb_only:
            depth_out.append(F.normalize(encoder(dep.to(device)), dim=1).cpu().numpy())
        if i % 50 == 0 or i + 1 == len(loader):
            print(f'Encoding batch {i + 1}/{len(loader)}', flush=True)
    return np.concatenate(rgb_out), None if rgb_only else np.concatenate(depth_out)


def extract(manifest, work, device='auto', batch_size=16, pretrained=True):
    work = Path(work)
    dest = work / 'features.npz'
    if dest.exists():
        raise FileExistsError(f'{dest} already exists; reuse it or choose a new --work folder.')
    seed_everything(manifest['seed'])
    device = choose_device(device)
    encoder = make_encoder(device, pretrained)
    rgb, depth = encode(manifest, encoder, device, batch_size)
    rows = manifest['rows']
    np.savez_compressed(dest, rgb=rgb, depth=depth, labels=np.array([r['label'] for r in rows]),
                        splits=np.array([r['split'] for r in rows]), ids=np.array([r['id'] for r in rows]),
                        fingerprint=fingerprint(manifest), pretrained=pretrained,
                        encoder='resnet18_imagenet1k_v1' if pretrained else 'random_resnet18_smoke_test')
    save_json(work / 'feature_config.json', dict(batch_size=batch_size, pretrained=pretrained))
    print(f'Saved features: {dest}', flush=True)


def load_features(work, manifest):
    with np.load(Path(work) / 'features.npz', allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    if str(data['fingerprint']) != fingerprint(manifest):
        raise ValueError('Feature cache does not match manifest. Use a new work directory.')
    if list(data['ids']) != [r['id'] for r in manifest['rows']]:
        raise ValueError('Feature/sample alignment mismatch.')
    if list(data['labels']) != [r['label'] for r in manifest['rows']] or list(data['splits']) != [r['split'] for r in manifest['rows']]:
        raise ValueError('Feature labels/splits do not match manifest.')
    if data['rgb'].shape != data['depth'].shape or data['rgb'].shape[0] != len(manifest['rows']):
        raise ValueError('Feature shape mismatch.')
    if not np.isfinite(data['rgb']).all() or not np.isfinite(data['depth']).all():
        raise ValueError('Nonfinite cached features.')
    return data




def squared_distances(x, reference):
    x, reference = np.asarray(x, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    return np.maximum((x*x).sum(1)[:, None] + (reference*reference).sum(1)[None, :] - 2*x@reference.T, 0.)


def training_scale(distance):
    values = distance[np.triu_indices(len(distance), 1)]
    positive = values[values > 1e-12]
    return float(np.median(positive)) if len(positive) else 1.


def additive_kernel(rgb, depth, reference_rgb, reference_depth, scales, gamma):
    """两个模态的 RBF 核固定等权相加；尺度仅由训练样本估计。"""
    kr = np.exp(-gamma * squared_distances(rgb, reference_rgb) / scales[0])
    kd = np.exp(-gamma * squared_distances(depth, reference_depth) / scales[1])
    return (kr + kd) / 2
