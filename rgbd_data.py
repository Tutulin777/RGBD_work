"""数据下载、RGB/Depth 配对、按物体实例划分；此模块不依赖 PyTorch。"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

BASE_URL = 'https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_eval/'
PATTERN = re.compile(r'^(?P<category>.+)_(?P<instance>\d+)_(?P<video>\d+)_(?P<frame>\d+)_crop\.png$')


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def download(data_dir):
    """下载到 .part；下载完成后改名。已存在的完整文件不重复下载。"""
    target = Path(data_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name in ['README.txt', 'testinstance_ids.txt', 'rgbd-dataset_eval.zip']:
        dest = target / name
        if dest.exists():
            print(f'Already exists: {dest}', flush=True)
            continue
        part = dest.with_suffix(dest.suffix + '.part')
        print(f'Downloading {BASE_URL + name}', flush=True)
        request = urllib.request.Request(BASE_URL + name, headers={'User-Agent': 'RGBD-Coursework/1.0'})
        with urllib.request.urlopen(request, timeout=90) as response, part.open('wb') as output:
            total = int(response.headers.get('Content-Length', 0))
            done, last = 0, 0
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                done += len(chunk)
                if done - last >= 32 * 1024 * 1024:
                    print(f'  {done / 2**20:.0f} / {total / 2**20:.0f} MiB', flush=True)
                    last = done
            if total and done != total:
                raise IOError(f'Incomplete download: {done} != {total}; rerun download.')
        part.replace(dest)
    marker = target / '.extracted_ok'
    if not marker.exists():
        (target / '.extracting').write_text('Extraction in progress. Rerun download if interrupted.\n', encoding='utf-8')
        with zipfile.ZipFile(target / 'rgbd-dataset_eval.zip') as archive:
            root = target.resolve()
            for member in archive.infolist():
                resolved = (root / member.filename).resolve()
                if not resolved.is_relative_to(root):
                    raise ValueError(f'Unsafe zip member: {member.filename}')
            print('Extracting archive ...', flush=True)
            archive.extractall(root)
        marker.write_text('Extraction completed.\n', encoding='utf-8')
        (target / '.extracting').unlink(missing_ok=True)
    print(f'Data ready: {target.resolve()}', flush=True)


def scan_pairs(root):
    root = Path(root).resolve()
    rows, missing = [], []
    for rgb in sorted(root.rglob('*_crop.png')):
        match = PATTERN.match(rgb.name)
        if not match:
            continue
        info = match.groupdict()
        stem = rgb.name[:-len('_crop.png')]
        depth = rgb.with_name(stem + '_depthcrop.png')
        if not depth.is_file():
            missing.append(str(rgb.relative_to(root)))
            continue
        rows.append(dict(id=stem, category=info['category'],
                         instance=f"{info['category']}_{info['instance']}",
                         video=int(info['video']), frame=int(info['frame']),
                         rgb=rgb.relative_to(root).as_posix(), depth=depth.relative_to(root).as_posix()))
    if not rows:
        raise ValueError(f'No paired *_crop.png / *_depthcrop.png files under {root}. Run download first.')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample IDs: choose a data root containing only one dataset copy.')
    return rows, missing


def split_instances(rows, seed=42, max_classes=10, max_frames=90, categories=None):
    """课程自定义协议：每类 1 个实例测试、1 个实例验证，其余训练。"""
    if max_classes < 0 or max_frames < 0:
        raise ValueError('max_classes/max_frames must be >= 0 (0 means all).')
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row['category']][row['instance']].append(row)
    eligible = sorted(c for c, objects in grouped.items() if len(objects) >= 3)
    if categories:
        selected = sorted(set(categories))
        bad = set(selected) - set(eligible)
        if bad:
            raise ValueError(f'Unknown categories or fewer than 3 instances: {sorted(bad)}')
    else:
        selected = eligible[:max_classes] if max_classes else eligible
    if len(selected) < 2:
        raise ValueError('Classification requires at least 2 categories, each with >= 3 instances.')
    rng = np.random.default_rng(seed)
    output = []
    for label, category in enumerate(selected):
        instances = sorted(grouped[category])
        rng.shuffle(instances)
        for index, instance in enumerate(instances):
            split = 'test' if index == 0 else 'val' if index == 1 else 'train'
            frames = sorted(grouped[category][instance], key=lambda r: (r['video'], r['frame']))
            if max_frames and len(frames) > max_frames:
                positions = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
                frames = [frames[i] for i in positions]
            output.extend(dict(row, label=label, split=split) for row in frames)
    assert_disjoint(output)
    return output, selected


def assert_disjoint(rows):
    splits = {s: {r['instance'] for r in rows if r['split'] == s} for s in ('train', 'val', 'test')}
    if any(not v for v in splits.values()):
        raise ValueError('An empty train/val/test split was found.')
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if splits[a] & splits[b]:
            raise ValueError(f'Object-instance leakage between {a} and {b}.')


def estimate_depth_range(rows, root, seed):
    """仅从训练图像估计 1%/99% 深度分位数；0 为缺失值，不参与统计。"""
    train = [r for r in rows if r['split'] == 'train']
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(train), min(128, len(train)), replace=False)
    values = []
    for index in chosen:
        with Image.open(Path(root) / train[index]['depth']) as im:
            arr = np.asarray(im)
        if arr.ndim != 2 or arr.dtype.kind not in 'ui' or arr.dtype.itemsize < 2:
            raise ValueError('Expected original single-channel 16-bit depth, not a colored preview.')
        valid = arr[arr > 0].reshape(-1)
        if valid.size:
            values.append(valid[::max(1, valid.size // 4096)])
    if not values:
        raise ValueError('No valid depth in sampled training images.')
    low, high = np.percentile(np.concatenate(values), [1, 99])
    if high <= low:
        high = low + 1
    return {'low_mm': float(low), 'high_mm': float(high), 'estimated_from': 'train only'}


def prepare(root, output, seed=42, max_classes=10, max_frames=90, categories=None):
    if (Path(root) / '.extracting').exists():
        raise RuntimeError('Data extraction is incomplete. Wait for download to finish or rerun download.')
    output = Path(output)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        raise FileExistsError(f'{manifest_path} already exists. Choose a NEW --work folder to preserve your split/results.')
    all_rows, missing = scan_pairs(root)
    rows, classes = split_instances(all_rows, seed, max_classes, max_frames, categories)
    depth = estimate_depth_range(rows, root, seed)
    # 验证所选图像能读取、尺寸配对；不读取标签以外的测试统计来训练。
    for row in rows:
        with Image.open(Path(root) / row['rgb']) as rgb, Image.open(Path(root) / row['depth']) as dep:
            if rgb.size != dep.size or len(dep.getbands()) != 1:
                raise ValueError(f'Mismatched RGB/depth pair: {row["id"]}')
            rgb.verify()
            dep.verify()
    manifest = dict(schema=1, root=str(Path(root).resolve()), seed=seed, classes=classes,
                    protocol='custom instance split: 1 test + 1 validation instance per class; remaining train',
                    max_frames_per_instance=max_frames, depth=depth, rows=rows)
    save_json(manifest_path, manifest)
    summary = dict(total_pairs_found=len(all_rows), missing_depth=len(missing), missing_examples=missing[:20],
                   selected_classes=classes, selected_samples=len(rows),
                   splits={s: dict(frames=sum(r['split'] == s for r in rows),
                                   instances=len({r['instance'] for r in rows if r['split'] == s}))
                           for s in ('train', 'val', 'test')},
                   per_class={c: dict(Counter(r['split'] for r in rows if r['category'] == c)) for c in classes})
    save_json(output / 'data_summary.json', summary)
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return manifest


def stable_fraction(key, seed):
    digest = hashlib.sha256(f'{seed}:{key}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big') / 2**64


def preprocess(row, root, depth_range, size=224, occlusion=0., seed=42):
    with Image.open(Path(root) / row['rgb']) as im:
        rgb = np.array(im.convert('RGB').resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    with Image.open(Path(root) / row['depth']) as im:
        raw = np.asarray(im).astype(np.float32)
    if raw.ndim != 2:
        raise ValueError('Depth must be single channel.')
    low, high = depth_range['low_mm'], depth_range['high_mm']
    if high <= low:
        raise ValueError('Invalid depth normalization range.')
    # 保留 0 表示缺失；有效深度映射到 [0.05, 1]，最近邻避免混入无效值。
    dep = np.where(raw > 0, 0.05 + .95 * np.clip((raw - low) / (high - low), 0, 1), 0).astype(np.float32)
    dep = np.asarray(Image.fromarray(dep).resize((size, size), Image.Resampling.NEAREST))
    if not 0 <= occlusion <= 1:
        raise ValueError('Occlusion must be between 0 and 1.')
    if occlusion:
        side = min(size, max(1, round(size * np.sqrt(occlusion))))
        y = int(stable_fraction(row['id'] + ':y', seed) * (size - side + 1))
        x = int(stable_fraction(row['id'] + ':x', seed) * (size - side + 1))
        rgb[y:y + side, x:x + side] = 0
    return rgb, np.repeat(dep[..., None], 3, axis=2)


def preview(manifest, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows, seen = [], set()
    # 优先展示不同类别；类别不足时选不同物体，总计最多 6 组。
    for key in ['category', 'instance', 'id']:
        used = set()
        for row in manifest['rows']:
            if len(rows) == 6:
                break
            if row['id'] not in seen and row[key] not in used:
                rows.append(row)
                seen.add(row['id'])
                used.add(row[key])
        if len(rows) == 6:
            break
    fig, axes = plt.subplots(len(rows), 2, figsize=(9, 3 * len(rows)), squeeze=False)
    records = []
    for i, row in enumerate(rows):
        with Image.open(Path(manifest['root']) / row['rgb']) as im:
            rgb = np.array(im.convert('RGB'))
        with Image.open(Path(manifest['root']) / row['depth']) as im:
            dep = np.array(im)
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f'{row["id"]}\nRGB | {row["split"]} | {rgb.shape} {rgb.dtype}', fontsize=9)
        plot = axes[i, 1].imshow(np.ma.masked_equal(dep, 0), cmap='viridis')
        axes[i, 1].set_title(f'Depth | {dep.shape} {dep.dtype}; 0 = missing', fontsize=9)
        fig.colorbar(plot, ax=axes[i, 1], label='Depth (mm)', shrink=.8)
        for ax in axes[i]:
            ax.axis('off')
        records.append(dict(row, rgb_shape=list(rgb.shape), rgb_dtype=str(rgb.dtype),
                            depth_shape=list(dep.shape), depth_dtype=str(dep.dtype)))
    fig.tight_layout()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / 'samples.png', dpi=160)
    plt.close(fig)
    save_json(output / 'sample_fields.json', records)
