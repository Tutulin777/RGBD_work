"""统一入口：RGB、Depth、概率平均、加性多核融合。"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('TORCH_HOME', str(ROOT/'cache'/'torch'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'cache'/'matplotlib'))


def doctor():
    import importlib.metadata
    import torch
    from torchvision.models import resnet18
    print('Python:',sys.version)
    for name in ('torch','torchvision','numpy','Pillow','scikit-learn','matplotlib','joblib','pytest'):
        print(f'{name}: {importlib.metadata.version(name)}')
    print('CUDA available:',torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:',torch.cuda.get_device_name(0))
        model=resnet18(weights=None).eval().cuda()
        with torch.inference_mode():
            output=model(torch.zeros(1,3,224,224,device='cuda'))
        torch.cuda.synchronize()
        print('CUDA ResNet18 forward: PASS',tuple(output.shape))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('doctor',help='Check environment and CUDA')
    for command in ('download','prepare','extract','train','evaluate','predict','verify','all'):
        p=commands.add_parser(command)
        if command != 'download':
            p.add_argument('--work',type=Path,default=ROOT/'runs'/'main')
        if command in ('download','prepare','all'):
            p.add_argument('--data',type=Path,default=ROOT/'data')
        if command in ('prepare','all'):
            p.add_argument('--seed',type=int,default=42)
            p.add_argument('--max-classes',type=int,default=10)
            p.add_argument('--max-frames',type=int,default=90)
        if command in ('extract','evaluate','predict','verify','all'):
            p.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
        if command in ('extract','all'):
            p.add_argument('--batch-size',type=int,default=16)
        if command == 'evaluate':
            p.add_argument('--batch-size',type=int,help='Default: reuse feature extraction batch size')
            p.add_argument('--levels',type=float,nargs='+',default=[0.,.1,.3,.5])
        if command in ('train','all'):
            p.add_argument('--epochs',type=int,default=40)
        if command == 'predict':
            group=p.add_mutually_exclusive_group()
            group.add_argument('--sample-id')
            group.add_argument('--rgb',type=Path)
            p.add_argument('--depth',type=Path)
        if command in ('verify','all'):
            p.add_argument('--raw-count',type=int,default=6 if command == 'all' else 0)
        if command == 'all':
            p.add_argument('--skip-download',action='store_true')
    args=parser.parse_args()
    if args.command == 'doctor':
        doctor()
        return
    from rgbd_data import download,load_json,save_json,prepare,preview
    if args.command == 'download':
        download(args.data)
        return
    work=args.work.resolve()
    if args.command == 'all':
        config=dict(data=str(args.data.resolve()),seed=args.seed,max_classes=args.max_classes,
                    max_frames=args.max_frames,epochs=args.epochs,batch_size=args.batch_size)
        config_path=work/'run_config.json'
        if config_path.exists():
            if load_json(config_path) != config:
                raise ValueError('Configuration changed. Choose a new --work directory.')
        elif work.exists() and any(work.iterdir()):
            raise ValueError('Use a new --work directory for all, or continue the manual commands.')
        save_json(config_path,config)
        if not args.skip_download:
            download(args.data)
        if not (work/'manifest.json').exists():
            manifest=prepare(args.data,work,args.seed,args.max_classes,args.max_frames)
            preview(manifest,work/'figures')
        manifest=load_json(work/'manifest.json')
        from rgbd_models import extract
        from rgbd_experiment import train,evaluate
        from verify_results import verify
        if not (work/'features.npz').exists():
            extract(manifest,work,args.device,args.batch_size)
        if not all((work/name).exists() for name in ('model.pt','fusion.joblib','training.json')):
            train(manifest,work,args.epochs)
        evaluate(manifest,work,args.device,args.batch_size)
        verify(manifest,work,args.raw_count,args.device)
        return
    if args.command == 'prepare':
        manifest=prepare(args.data,work,args.seed,args.max_classes,args.max_frames)
        preview(manifest,work/'figures')
        return
    manifest=load_json(work/'manifest.json')
    if args.command == 'extract':
        from rgbd_models import extract
        extract(manifest,work,args.device,args.batch_size)
    elif args.command == 'train':
        from rgbd_experiment import train
        train(manifest,work,args.epochs)
    elif args.command == 'evaluate':
        from rgbd_experiment import evaluate
        evaluate(manifest,work,args.device,args.batch_size,args.levels)
    elif args.command == 'verify':
        from verify_results import verify
        verify(manifest,work,args.raw_count,args.device)
    elif args.command == 'predict':
        from rgbd_experiment import predict_image
        result=predict_image(manifest,work,args.device,args.rgb,args.depth,args.sample_id)
        print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
