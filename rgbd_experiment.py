"""四种方法的训练、推理、统一评估和绘图。"""
from __future__ import annotations

import copy
import csv
import hashlib
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC

from rgbd_data import load_json, save_json, stable_fraction
from rgbd_models import (additive_kernel, choose_device, encode, fingerprint,
                         load_features, make_encoder, seed_everything,
                         squared_distances, training_scale)

METHODS = ('rgb', 'depth', 'average', 'additive_kernel')
CS = (.1, 1., 10., 100.)
GAMMAS = (.5, 1., 2.)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, prediction, nclass):
    return dict(accuracy=float(accuracy_score(y, prediction)),
                macro_f1=float(f1_score(y, prediction, labels=np.arange(nclass),
                                       average='macro', zero_division=0)))


def fit_head(x, y, train_indices, val_indices, epochs, seed, nclass):
    """线性分类头，按验证损失选择最佳轮次。"""
    torch.manual_seed(seed)
    head = nn.Linear(x.shape[1], nclass)
    optimizer = torch.optim.Adam(head.parameters(), lr=.003, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best_loss, best_state, best_epoch, history = float('inf'), None, 0, []
    for epoch in range(1, epochs + 1):
        head.train()
        order, total = rng.permutation(len(train_indices)), 0.
        for start in range(0, len(order), 128):
            ix = train_indices[order[start:start + 128]]
            optimizer.zero_grad()
            loss = F.cross_entropy(head(x[ix]), y[ix])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ix)
        head.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(head(x[val_indices]), y[val_indices]).item()
        history.append(dict(epoch=epoch, train_loss=total/len(train_indices), val_loss=val_loss))
        if val_loss < best_loss:
            best_loss, best_epoch, best_state = val_loss, epoch, copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    return head.eval(), dict(best_epoch=best_epoch, best_val_loss=best_loss, history=history)


def train(manifest, work, epochs=40):
    work = Path(work)
    if epochs < 1:
        raise ValueError('epochs must be positive')
    if any((work / name).exists() for name in ('model.pt', 'fusion.joblib', 'training.json')):
        raise FileExistsError('Training outputs exist. Choose a new --work directory.')
    data = load_features(work, manifest)
    if not bool(data['pretrained']):
        raise ValueError('Formal experiments require pretrained encoder features.')
    seed = manifest['seed']
    seed_everything(seed)
    tr, va = [np.flatnonzero(data['splits'] == s) for s in ('train', 'val')]
    y = torch.as_tensor(data['labels'], dtype=torch.long)
    heads, histories = {}, {}
    for name in ('rgb', 'depth'):
        print(f'Training {name} linear head ...', flush=True)
        head, histories[name] = fit_head(torch.as_tensor(data[name], dtype=torch.float32),
                                         y, tr, va, epochs, seed, len(manifest['classes']))
        heads[name] = head.state_dict()
        print(f"  Best epoch {histories[name]['best_epoch']}", flush=True)

    r, d = [np.asarray(data[k][tr], dtype=np.float64) for k in ('rgb', 'depth')]
    vr, vd = [np.asarray(data[k][va], dtype=np.float64) for k in ('rgb', 'depth')]
    dr, dd = squared_distances(r, r), squared_distances(d, d)
    vdr, vdd = squared_distances(vr, r), squared_distances(vd, d)
    scales = [training_scale(dr), training_scale(dd)]
    trials, best_record, best_model = [], None, None
    for gamma in GAMMAS:
        kt = (np.exp(-gamma*dr/scales[0]) + np.exp(-gamma*dd/scales[1])) / 2
        kv = (np.exp(-gamma*vdr/scales[0]) + np.exp(-gamma*vdd/scales[1])) / 2
        for c in CS:
            model = SVC(C=c, kernel='precomputed', cache_size=256,
                        decision_function_shape='ovr', random_state=seed)
            model.fit(kt, data['labels'][tr])
            record = dict(C=c, gamma=gamma,
                          **metrics(data['labels'][va], model.predict(kv), len(manifest['classes'])))
            trials.append(record)
            if best_record is None or (record['accuracy'], record['macro_f1']) > (
                    best_record['accuracy'], best_record['macro_f1']):
                best_record, best_model = record, model
    print(f'Additive kernel: validation-selected {best_record}', flush=True)
    identity = dict(schema=3, fingerprint=fingerprint(manifest), classes=manifest['classes'])
    torch.save(dict(identity, **heads, dim=data['rgb'].shape[1], pretrained=True,
                    seed=seed, epochs=epochs), work / 'model.pt')
    joblib.dump(dict(identity, model=best_model, rgb=r, depth=d, scales=scales,
                     selected=best_record, train_ids=data['ids'][tr]), work / 'fusion.joblib', compress=3)
    source_names = ('main.py', 'rgbd_data.py', 'rgbd_models.py', 'rgbd_experiment.py', 'verify_results.py')
    save_json(work / 'training.json', dict(
        methods=METHODS, seed=seed, epochs=epochs, head_lr=.003, head_batch_size=128,
        weight_decay=1e-4, heads=histories,
        fusion=dict(C_grid=CS, gamma_grid=GAMMAS, trials=trials, selected=best_record,
                    train_ids=data['ids'][tr].tolist(), val_ids=data['ids'][va].tolist(),
                    selection='max validation accuracy, then macro-F1; ties retain first grid candidate'),
        model_sha256={name: digest(work/name) for name in ('model.pt', 'fusion.joblib')},
        source_sha256={name: digest(Path(__file__).parent/name) for name in source_names},
        torch=str(torch.__version__), sklearn=__import__('sklearn').__version__,
        protocol=manifest['protocol'],
        missing_policy='Both fusion methods fall back to the SAME retained RGB linear classifier.'))


def load_models(work, manifest, data=None):
    work = Path(work)
    training = load_json(work/'training.json')
    if tuple(training['methods']) != METHODS:
        raise ValueError('Unexpected method set; use a four-method experiment.')
    for name, expected in training['model_sha256'].items():
        if digest(work/name) != expected:
            raise ValueError(f'Model checksum mismatch: {name}')
    ck = torch.load(work/'model.pt', map_location='cpu', weights_only=True)
    fusion = joblib.load(work/'fusion.joblib')
    for item in (ck, fusion):
        if item['schema'] != 3 or item['fingerprint'] != fingerprint(manifest) or item['classes'] != manifest['classes']:
            raise ValueError('Model and manifest do not match.')
    if data is not None and ck['pretrained'] != bool(data['pretrained']):
        raise ValueError('Encoder and model weights do not match.')
    heads = []
    for name in ('rgb', 'depth'):
        head = nn.Linear(ck['dim'], len(ck['classes']))
        head.load_state_dict(ck[name])
        heads.append(head.eval())
    return ck, heads, fusion


@torch.inference_mode()
def predict_features(rgb, depth, heads, fusion, missing=None):
    """返回四种方法的类别，以及前三种方法的概率；SVM不伪装成概率。"""
    rgb, depth = np.asarray(rgb), np.asarray(depth)
    if rgb.shape != depth.shape or rgb.ndim != 2:
        raise ValueError('Expected matching two-dimensional RGB/depth features.')
    mask = np.zeros(len(rgb), dtype=bool) if missing is None else np.asarray(missing, dtype=bool)
    if mask.shape != (len(rgb),):
        raise ValueError('Invalid missing-modality mask.')
    r = torch.as_tensor(rgb, dtype=torch.float32)
    d = torch.as_tensor(depth, dtype=torch.float32).clone()
    d[mask] = 0
    rp, dp = [head(x).softmax(1).numpy() for head, x in zip(heads, (r, d))]
    dp[mask] = 1 / dp.shape[1]
    average = (rp + dp) / 2
    average[mask] = rp[mask]
    probabilities = dict(rgb=rp, depth=dp, average=average)
    predictions = {name: p.argmax(1) for name, p in probabilities.items()}
    fused = predictions['rgb'].copy()
    complete = ~mask
    if complete.any():
        query = additive_kernel(rgb[complete], depth[complete], fusion['rgb'], fusion['depth'],
                                fusion['scales'], fusion['selected']['gamma'])
        fused[complete] = fusion['model'].predict(query)
    predictions['additive_kernel'] = fused
    return predictions, probabilities


def evaluate(manifest, work, device='auto', batch_size=None, levels=(0., .1, .3, .5)):
    work = Path(work)
    if any(not 0 <= v <= 1 for v in levels):
        raise ValueError('Corruption levels must be in [0, 1].')
    if batch_size is None:
        batch_size = load_json(work/'feature_config.json')['batch_size']
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    data = load_features(work, manifest)
    _, heads, fusion = load_models(work, manifest, data)
    seed_everything(manifest['seed'])
    te = np.flatnonzero(data['splits'] == 'test')
    r, d, y, ids = [data[k][te] for k in ('rgb', 'depth', 'labels', 'ids')]
    rows = [manifest['rows'][i] for i in te]
    result = work/'results'
    result.mkdir(exist_ok=True)
    (result/'verification.json').unlink(missing_ok=True)
    save_json(result/'evaluation.json', dict(status='RUNNING'))
    records, saved_predictions, saved_probs, saved_rgb, masks, scenarios, strengths = [], [], [], [], [], [], []

    def measure(scenario, level, rgb, missing=None):
        mask = np.zeros(len(y), dtype=bool) if missing is None else missing
        predictions, probabilities = predict_features(rgb, d, heads, fusion, mask)
        for method in METHODS:
            records.append(dict(scenario=scenario, level=float(level), method=method, samples=len(y),
                missing_count=int(mask.sum()), **metrics(y, predictions[method], len(manifest['classes']))))
        saved_predictions.append(np.stack([predictions[m] for m in METHODS]))
        saved_probs.append(np.stack([probabilities[m] for m in METHODS[:3]]))
        saved_rgb.append(rgb)
        masks.append(mask)
        scenarios.append(scenario)
        strengths.append(float(level))
        print(f'Evaluated {scenario} {level:g}', flush=True)
        if scenario == 'clean':
            for record in records:
                print(f"  {record['method']}: {record['accuracy']:.2%}", flush=True)

    measure('clean', 0., r)
    nonzero = sorted(set(float(v) for v in levels if v > 0))
    if nonzero:
        dev = choose_device(device)
        seed_everything(manifest['seed'])
        encoder = make_encoder(dev)
        for level in nonzero:
            altered, _ = encode(manifest, encoder, dev, batch_size, rows=rows, occlusion=level, rgb_only=True)
            measure('rgb_occlusion', level, altered)
        del encoder
        order = np.argsort([stable_fraction(row['id']+':missing', manifest['seed']) for row in rows])
        for level in nonzero:
            mask = np.zeros(len(y), dtype=bool)
            mask[order[:round(level*len(y))]] = True
            measure('depth_missing', level, r, mask)
    np.savez_compressed(result/'predictions.npz', ids=ids, labels=y, methods=np.array(METHODS),
        scenarios=np.array(scenarios), levels=np.array(strengths), predictions=np.stack(saved_predictions),
        probabilities=np.stack(saved_probs), rgb_inputs=np.stack(saved_rgb), depth_inputs=d,
        depth_missing=np.stack(masks))
    with (result/'metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    draw_results(records, saved_predictions[0], y, manifest['classes'], result)
    save_json(result/'evaluation.json', dict(status='COMPLETE', methods=METHODS,
        conditions=len(scenarios), samples=len(y), levels=[0., *nonzero], batch_size=batch_size,
        training_sha256=digest(work/'training.json'),
        missing_policy='Both fusion methods use the retained RGB linear classifier when depth is missing.'))
    return records


def draw_results(records, clean_predictions, y, classes, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    labels = ['RGB', 'Depth', 'Probability mean', 'Additive kernel']
    colors = ['#767676', '#B64342', '#42949E', '#0F4D92']
    clean = {r['method']: r['accuracy'] for r in records if r['scenario'] == 'clean'}
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    bars = ax.bar(labels, [100*clean[m] for m in METHODS], color=colors)
    ax.bar_label(bars, fmt='%.2f', padding=4)
    ax.set(ylabel='Accuracy (%)', ylim=(0, 108), title='Clean test: identical instance split')
    fig.savefig(result/'comparison.png', dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for ax, scenario, title in zip(axes, ('rgb_occlusion', 'depth_missing'), ('RGB occlusion', 'Depth modality missing')):
        for method, label, color, marker in zip(METHODS, labels, colors, ('o','s','^','D')):
            points = [(0., clean[method])] + sorted((r['level'], r['accuracy']) for r in records
                                                  if r['scenario'] == scenario and r['method'] == method)
            ax.plot([100*x for x,_ in points], [100*v for _,v in points], label=label,
                    color=color, marker=marker, linewidth=2)
        ax.set(title=title, xlabel='Area (%)' if scenario == 'rgb_occlusion' else 'Samples (%)',
               ylabel='Accuracy (%)', ylim=(0, 105))
        ax.grid(alpha=.2)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='outside lower center', ncol=4, frameon=False)
    fig.savefig(result/'robustness.png', dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), layout='constrained')
    for ax, prediction, label in zip(axes.flat, clean_predictions, labels):
        cm = confusion_matrix(y, prediction, labels=np.arange(len(classes)))
        picture = ax.imshow(cm, cmap='Blues')
        ax.set(title=label, xlabel='Predicted', ylabel='True', xticks=range(len(classes)),
               yticks=range(len(classes)), xticklabels=classes, yticklabels=classes)
        plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=8)
        fig.colorbar(picture, ax=ax, shrink=.8)
    fig.savefig(result/'confusion_matrices.png', dpi=200)
    plt.close(fig)


def predict_image(manifest, work, device='auto', rgb_path=None, depth_path=None, sample_id=None):
    from PIL import Image
    if (rgb_path is None) != (depth_path is None):
        raise ValueError('Supply both --rgb and --depth.')
    if rgb_path is not None and sample_id is not None:
        raise ValueError('Use either external image paths or --sample-id.')
    if rgb_path is not None:
        rgb_path, depth_path = Path(rgb_path).resolve(), Path(depth_path).resolve()
        with Image.open(rgb_path) as a, Image.open(depth_path) as b:
            dep = np.asarray(b)
            if a.size != b.size or dep.ndim != 2 or dep.dtype.kind not in 'ui' or dep.dtype.itemsize < 2:
                raise ValueError('Expected matching image sizes and a single-channel integer depth image in mm.')
        row = dict(id=rgb_path.stem, rgb=str(rgb_path), depth=str(depth_path))
        truth = None
    else:
        candidates = [r for r in manifest['rows'] if r['id'] == sample_id] if sample_id else [r for r in manifest['rows'] if r['split'] == 'test']
        if not candidates:
            raise ValueError(f'Unknown sample ID: {sample_id}')
        row = candidates[0]
        truth = manifest['classes'][row['label']]
    _, heads, fusion = load_models(work, manifest)
    seed_everything(manifest['seed'])
    dev = choose_device(device)
    encoder = make_encoder(dev)
    r, d = encode(manifest, encoder, dev, batch_size=1, rows=[row])
    predictions, _ = predict_features(r, d, heads, fusion)
    return dict(sample_id=row['id'], truth_if_known=truth,
                predictions={m: manifest['classes'][int(predictions[m][0])] for m in METHODS})

