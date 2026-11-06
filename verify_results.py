"""独立重算四种方法的指标，检查划分、选模、缺失策略和模型回放。"""
from pathlib import Path
import csv
import numpy as np

from rgbd_data import assert_disjoint, estimate_depth_range, load_json, save_json, stable_fraction
from rgbd_models import (additive_kernel, load_features, squared_distances, training_scale,
                         choose_device, make_encoder, encode, seed_everything)
from rgbd_experiment import METHODS, CS, GAMMAS, digest, load_models, predict_features


def independent_metrics(y, prediction, nclass):
    if y.shape != prediction.shape or np.any(prediction < 0) or np.any(prediction >= nclass):
        raise ValueError('Invalid predicted labels')
    matrix = np.zeros((nclass, nclass), dtype=np.int64)
    np.add.at(matrix, (y, prediction), 1)
    den = matrix.sum(0) + matrix.sum(1)
    f1 = np.divide(2*np.diag(matrix), den, out=np.zeros(nclass), where=den > 0)
    return float(np.trace(matrix)/len(y)), float(f1.mean())


def verify(manifest, work, raw_count=0, device='auto'):
    work = Path(work)
    result = work/'results'
    checks = []
    def require(condition, message):
        if not bool(condition):
            raise ValueError(message)
    try:
        require(raw_count >= 0, 'raw_count cannot be negative')
        assert_disjoint(manifest['rows'])
        require(len({r['id'] for r in manifest['rows']}) == len(manifest['rows']), 'Duplicate sample IDs')
        for row in manifest['rows']:
            require(manifest['classes'][row['label']] == row['category'], 'Wrong category label')
            require(all((Path(manifest['root'])/row[k]).is_file() for k in ('rgb','depth')), 'Missing raw image')
        depth_range = estimate_depth_range(manifest['rows'], manifest['root'], manifest['seed'])
        require(depth_range == manifest['depth'], 'Depth normalization is not train-only')
        data = load_features(work, manifest)
        _, heads, fusion = load_models(work, manifest, data)
        training = load_json(work/'training.json')
        tr, va, te = [np.flatnonzero(data['splits'] == s) for s in ('train','val','test')]
        require(training['fusion']['train_ids'] == data['ids'][tr].tolist(), 'Wrong training reference IDs')
        require(training['fusion']['val_ids'] == data['ids'][va].tolist(), 'Wrong validation IDs')
        require(np.array_equal(fusion['train_ids'], data['ids'][tr]), 'Wrong stored kernel reference IDs')
        for i, name in enumerate(('rgb','depth')):
            require(np.array_equal(fusion[name], data[name][tr]), 'Kernel reference is not training-only')
            require(abs(training_scale(squared_distances(fusion[name], fusion[name]))-fusion['scales'][i]) < 1e-12,
                    'Wrong training-only kernel scale')
            hist = training['heads'][name]
            chosen = min(hist['history'], key=lambda item: item['val_loss'])
            require(chosen['epoch'] == hist['best_epoch'] and chosen['val_loss'] == hist['best_val_loss'],
                    'Wrong validation-best head epoch')
        trials = training['fusion']['trials']
        require(len(trials) == len(CS)*len(GAMMAS), 'Incomplete SVM search')
        require([(t['gamma'], t['C']) for t in trials] == [(g,c) for g in GAMMAS for c in CS], 'Wrong SVM grid order')
        selected = max(trials, key=lambda t: (t['accuracy'], t['macro_f1']))
        require(selected == training['fusion']['selected'] == fusion['selected'], 'SVM is not validation-best')
        require(fusion['model'].C == selected['C'], 'Wrong SVM C')
        query = additive_kernel(data['rgb'][va], data['depth'][va], fusion['rgb'], fusion['depth'],
                                fusion['scales'], selected['gamma'])
        acc, f1 = independent_metrics(data['labels'][va], fusion['model'].predict(query), len(manifest['classes']))
        require(abs(acc-selected['accuracy']) < 1e-12 and abs(f1-selected['macro_f1']) < 1e-12,
                'Validation prediction replay differs')
        checks.append('Disjoint instances, train-only normalization/kernel references, validation-selected models')

        meta = load_json(result/'evaluation.json')
        require(meta['status'] == 'COMPLETE', 'Evaluation did not finish')
        require(meta['training_sha256'] == digest(work/'training.json'), 'Training changed after evaluation')
        require(tuple(meta['methods']) == METHODS, 'Wrong evaluation method set')
        with (result/'metrics.csv').open(encoding='utf-8-sig') as f:
            records = list(csv.DictReader(f))
        with np.load(result/'predictions.npz', allow_pickle=False) as archive:
            saved = {k: archive[k] for k in archive.files}
        conditions = [('clean',0.)] + [(s,float(v)) for s in ('rgb_occlusion','depth_missing')
                                      for v in meta['levels'] if v > 0]
        require(list(zip(saved['scenarios'], saved['levels'])) == conditions, 'Unexpected evaluation conditions')
        require(meta['conditions'] == len(conditions) and meta['samples'] == len(te), 'Wrong evaluation counts')
        require(tuple(saved['methods']) == METHODS, 'Unexpected prediction method order')
        keys = [(r['scenario'],float(r['level']),r['method']) for r in records]
        expected = {(s,v,m) for s,v in conditions for m in METHODS}
        require(len(keys) == len(expected) and set(keys) == expected, 'Missing/duplicate metrics')
        nclass, y = len(manifest['classes']), data['labels'][te]
        require(np.array_equal(saved['ids'], data['ids'][te]) and np.array_equal(saved['labels'], y), 'Test samples differ')
        require(saved['predictions'].shape == (len(conditions),4,len(te)), 'Wrong prediction dimensions')
        require(saved['probabilities'].shape == (len(conditions),3,len(te),nclass), 'Wrong probability dimensions')
        require(np.array_equal(saved['depth_inputs'], data['depth'][te]), 'Depth inputs changed')
        order = np.argsort([stable_fraction(str(i)+':missing', manifest['seed']) for i in saved['ids']])
        for i, (scenario, level) in enumerate(conditions):
            mask = np.zeros(len(y), dtype=bool)
            if scenario == 'depth_missing':
                mask[order[:round(level*len(y))]] = True
            require(np.array_equal(saved['depth_missing'][i], mask), 'Missing mask differs')
            rgb = saved['rgb_inputs'][i]
            if scenario != 'rgb_occlusion':
                require(np.array_equal(rgb, data['rgb'][te]), 'Clean RGB inputs changed')
            predictions, probabilities = predict_features(rgb, saved['depth_inputs'], heads, fusion, mask)
            for j, method in enumerate(METHODS):
                pred = saved['predictions'][i,j]
                require(np.array_equal(predictions[method], pred), 'Model prediction replay differs')
                row = records[keys.index((scenario,level,method))]
                acc, f1 = independent_metrics(y, pred, nclass)
                require(abs(acc-float(row['accuracy'])) < 1e-12 and abs(f1-float(row['macro_f1'])) < 1e-12,
                        'Metrics differ from independent calculation')
                require(int(row['samples']) == len(y) and int(row['missing_count']) == int(mask.sum()), 'Wrong sample count')
                if j < 3:
                    p = saved['probabilities'][i,j]
                    require(np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1),1,atol=1e-6), 'Invalid probability')
                    require(np.allclose(p, probabilities[method], atol=1e-7, rtol=1e-6), 'Probability replay differs')
            rp, dp, avg = saved['probabilities'][i]
            require(np.allclose(avg[~mask], (rp[~mask]+dp[~mask])/2, atol=1e-7), 'Wrong average formula')
            require(np.array_equal(avg[mask],rp[mask]), 'Average did not fall back to RGB')
            require(np.array_equal(saved['predictions'][i,3,mask], saved['predictions'][i,0,mask]),
                    'Kernel did not fall back to the same RGB classifier')
            if mask.any():
                require(np.allclose(dp[mask], 1/nclass), 'Unavailable depth must use uniform probability')
            if scenario == 'rgb_occlusion':
                require(np.array_equal(saved['predictions'][i,1],saved['predictions'][0,1]), 'RGB corruption changed depth-only')
            if scenario == 'depth_missing':
                require(np.array_equal(saved['predictions'][i,0],saved['predictions'][0,0]), 'Depth missing changed RGB-only')
        checks.append(f'{len(records)} metrics independently recalculated; all cached predictions and probabilities replayed')
        checks.append('Exactly four methods; both fusion methods use the same RGB fallback')
        raw_ids = []
        if raw_count:
            seed_everything(manifest['seed'])
            dev = choose_device(device)
            encoder = make_encoder(dev)
            positions = np.linspace(0,len(te)-1,min(raw_count,len(te)),dtype=int)
            selected_indices = te[positions]
            original_batch = load_json(work/'feature_config.json')['batch_size']
            test_rows = [manifest['rows'][k] for k in te]
            def replay(rows, selected_positions, batch_size, level=0.):
                rr, dd = {}, {}
                for start in sorted({int(k)//batch_size*batch_size for k in selected_positions}):
                    block = rows[start:start+batch_size]
                    r, d = encode(manifest, encoder, dev, batch_size, rows=block, occlusion=level)
                    for k in selected_positions:
                        if start <= k < start+len(block):
                            rr[int(k)], dd[int(k)] = r[int(k)-start], d[int(k)-start]
                return np.stack([rr[int(k)] for k in selected_positions]), np.stack([dd[int(k)] for k in selected_positions])
            rr, dd = replay(manifest['rows'], selected_indices, original_batch)
            np.testing.assert_allclose(rr,data['rgb'][selected_indices],atol=3e-5,rtol=2e-4)
            np.testing.assert_allclose(dd,data['depth'][selected_indices],atol=3e-5,rtol=2e-4)
            for i,(scenario,level) in enumerate(conditions):
                if scenario == 'depth_missing':
                    continue
                r = rr if scenario == 'clean' else replay(test_rows,positions,meta['batch_size'],level)[0]
                np.testing.assert_allclose(r,saved['rgb_inputs'][i,positions],atol=3e-5,rtol=2e-4)
                predictions, _ = predict_features(r,dd,heads,fusion)
                for j,method in enumerate(METHODS):
                    require(np.array_equal(predictions[method],saved['predictions'][i,j,positions]), 'Raw image prediction differs')
            raw_ids = data['ids'][selected_indices].tolist()
            checks.append(f'{len(raw_ids)} raw pairs replayed for clean and all RGB occlusion levels')
        report = dict(status='PASS', checks=checks, methods=METHODS, metric_rows=len(records), raw_sample_ids=raw_ids)
        save_json(result/'verification.json',report)
        print(f'Verification PASS: {len(records)} metric rows; {len(raw_ids)} raw pairs checked',flush=True)
        return report
    except Exception as exc:
        save_json(result/'verification.json',dict(status='FAIL',checks=checks,error=str(exc)))
        raise
