"""用人工构造的小图检查协议和计算，不把人工样本成绩当作实验结果。"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'cache' / 'matplotlib'))

import numpy as np
from PIL import Image
import pytest

from rgbd_data import (assert_disjoint, estimate_depth_range, prepare, preprocess,
                       scan_pairs, split_instances)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / 'data'
    root.mkdir()
    for label, category in enumerate(['apple', 'coffee_mug']):
        for instance in range(1, 5):
            folder = root / category / f'{category}_{instance}'
            folder.mkdir(parents=True)
            for frame in range(3):
                stem = f'{category}_{instance}_1_{frame}'
                rgb = np.zeros((24, 32, 3), dtype=np.uint8)
                rgb[..., label] = 100 + instance * 10
                depth = np.full((24, 32), 500 + instance * 10 + frame, dtype=np.uint16)
                depth[0] = 0
                Image.fromarray(rgb).save(folder / f'{stem}_crop.png')
                Image.fromarray(depth).save(folder / f'{stem}_depthcrop.png')
    return root


def test_pairing_and_instance_split(dataset):
    rows, missing = scan_pairs(dataset)
    assert len(rows) == 24 and missing == []
    first, classes = split_instances(rows, max_classes=0, max_frames=2)
    second, _ = split_instances(rows, max_classes=0, max_frames=2)
    assert first == second and classes == ['apple', 'coffee_mug']
    assert len(first) == 16
    assert_disjoint(first)
    leaking = first + [dict(first[0], split='train')]
    with pytest.raises(ValueError, match='leakage'):
        assert_disjoint(leaking)


def test_depth_statistics_do_not_use_test_data(dataset):
    rows, _ = scan_pairs(dataset)
    rows, _ = split_instances(rows)
    original = estimate_depth_range(rows, dataset, 42)
    for row in rows:
        if row['split'] != 'train':
            Image.fromarray(np.full((24, 32), 60000, dtype=np.uint16)).save(dataset / row['depth'])
    assert estimate_depth_range(rows, dataset, 42) == original


def test_incomplete_extraction_cannot_create_split(dataset, tmp_path):
    (dataset / '.extracting').write_text('in progress', encoding='utf-8')
    with pytest.raises(RuntimeError, match='extraction is incomplete'):
        prepare(dataset, tmp_path / 'must_not_exist')
    assert not (tmp_path / 'must_not_exist').exists()


def test_existing_split_is_not_silently_overwritten(dataset, tmp_path):
    work = tmp_path / 'fixed_split'
    prepare(dataset, work, seed=42)
    original = (work / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        prepare(dataset, work, seed=7)
    assert (work / 'manifest.json').read_bytes() == original


def test_preprocess_missing_and_corruption(dataset):
    rows, _ = scan_pairs(dataset)
    norm = {'low_mm': 500., 'high_mm': 600.}
    rgb, dep = preprocess(rows[0], dataset, norm, size=32)
    assert rgb.shape == dep.shape == (32, 32, 3)
    assert np.isfinite(dep).all() and dep.min() == 0 and dep.max() <= 1
    altered, same = preprocess(rows[0], dataset, norm, size=32, occlusion=.25)
    assert np.array_equal(dep, same)
    assert np.sum(np.any(rgb != altered, axis=-1)) == 256
    again, _ = preprocess(rows[0], dataset, norm, size=32, occlusion=.25)
    np.testing.assert_array_equal(altered, again)



from rgbd_models import fingerprint, additive_kernel, squared_distances, training_scale
from rgbd_experiment import METHODS, train, evaluate, load_models, predict_features
from verify_results import verify, independent_metrics
from rgbd_data import save_json, load_json


def write_features(manifest, work, rgb=None, depth=None):
    rows = manifest['rows']
    labels = np.array([r['label'] for r in rows])
    if rgb is None:
        rng = np.random.default_rng(3)
        rgb = rng.normal(0,.03,(len(rows),8)).astype(np.float32)
        rgb[np.arange(len(rows)),labels] += 1
        depth = rgb.copy()
    np.savez_compressed(work/'features.npz',rgb=rgb,depth=depth,labels=labels,
        splits=np.array([r['split'] for r in rows]),ids=np.array([r['id'] for r in rows]),
        fingerprint=fingerprint(manifest),pretrained=True)
    save_json(work/'feature_config.json',dict(batch_size=16,pretrained=True))
    return rgb,depth


@pytest.fixture
def feature_run(dataset,tmp_path):
    work = tmp_path/'run'
    manifest = prepare(dataset,work,max_classes=0)
    rgb,depth = write_features(manifest,work)
    return manifest,work,rgb,depth


def test_additive_kernel_is_positive_semidefinite():
    rng=np.random.default_rng(7)
    rgb,depth=rng.normal(size=(12,5)),rng.normal(size=(12,5))
    scales=[training_scale(squared_distances(x,x)) for x in (rgb,depth)]
    kernel=additive_kernel(rgb,depth,rgb,depth,scales,.5)
    np.testing.assert_allclose(kernel,kernel.T,atol=1e-12)
    np.testing.assert_allclose(np.diag(kernel),1,atol=1e-12)
    assert np.linalg.eigvalsh(kernel).min()>-1e-10


def test_exactly_four_methods_and_shared_missing_fallback(feature_run):
    manifest,work,rgb,depth=feature_run
    train(manifest,work,epochs=2)
    _,heads,fusion=load_models(work,manifest)
    mask=np.array([True,False,True])
    pred,prob=predict_features(rgb[:3],depth[:3],heads,fusion,mask)
    assert tuple(pred)==METHODS and tuple(prob)==METHODS[:3]
    np.testing.assert_array_equal(pred['average'][mask],pred['rgb'][mask])
    np.testing.assert_array_equal(pred['additive_kernel'][mask],pred['rgb'][mask])
    np.testing.assert_allclose(prob['depth'][mask],.5)
    np.testing.assert_allclose(prob['average'][~mask],(prob['rgb'][~mask]+prob['depth'][~mask])/2)
    changed=depth[:3].copy()
    changed[mask]=1e6
    after,_=predict_features(rgb[:3],changed,heads,fusion,mask)
    for method in METHODS:
        np.testing.assert_array_equal(pred[method],after[method])


def test_training_ignores_test_features_and_labels(feature_run,tmp_path):
    import copy
    import torch
    manifest,work,rgb,depth=feature_run
    train(manifest,work,epochs=2)
    second=tmp_path/'changed'
    second.mkdir()
    changed=copy.deepcopy(manifest)
    mask=np.array([r['split']=='test' for r in changed['rows']])
    for row in changed['rows']:
        if row['split']=='test':
            row['label']=1-row['label']
            row['category']=changed['classes'][row['label']]
    altered_rgb,altered_depth=rgb.copy(),depth.copy()
    altered_rgb[mask],altered_depth[mask]=1e6,-1e6
    save_json(second/'manifest.json',changed)
    write_features(changed,second,altered_rgb,altered_depth)
    train(changed,second,epochs=2)
    first_record,second_record=[load_json(p/'training.json') for p in (work,second)]
    assert first_record['heads']==second_record['heads']
    assert first_record['fusion']==second_record['fusion']
    a,b=[torch.load(p/'model.pt',weights_only=True) for p in (work,second)]
    for name in ('rgb','depth'):
        for key in a[name]:
            torch.testing.assert_close(a[name][key],b[name][key],rtol=0,atol=0)


def test_train_evaluate_verify_and_reject_tampered_metric(feature_run):
    import csv
    manifest,work,_,_=feature_run
    train(manifest,work,epochs=2)
    records=evaluate(manifest,work,levels=[0])
    assert len(records)==4 and {r['method'] for r in records}==set(METHODS)
    assert verify(manifest,work)['status']=='PASS'
    with np.load(work/'results/predictions.npz') as a:
        assert a['predictions'].shape[:2]==(1,4)
    for name in ('comparison.png','robustness.png','confusion_matrices.png'):
        assert (work/'results'/name).exists()
    records[0]['accuracy']=.123456
    with (work/'results/metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with pytest.raises(ValueError,match='independent calculation'):
        verify(manifest,work)
    assert load_json(work/'results/verification.json')['status']=='FAIL'


def test_existing_model_is_not_overwritten(feature_run):
    manifest,work,_,_=feature_run
    train(manifest,work,epochs=1)
    original=(work/'model.pt').read_bytes()
    with pytest.raises(FileExistsError):
        train(manifest,work,epochs=1)
    assert (work/'model.pt').read_bytes()==original


def test_independent_metric_formula():
    acc,f1=independent_metrics(np.array([0,0,1,1]),np.array([0,1,1,1]),2)
    assert acc==.75
    assert abs(f1-(2/3+.8)/2)<1e-12
