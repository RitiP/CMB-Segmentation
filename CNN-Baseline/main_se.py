import argparse
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from solver_patch import Solver
import datetime
import random


def seed_exps(manual_seed=100):
    if manual_seed > 0:
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        np.random.seed(manual_seed)
        random.seed(manual_seed)
        os.environ['PYTHONHASHSEED'] = str(manual_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(manual_seed)
            torch.cuda.manual_seed_all(manual_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    
def print_args(args):
    for k in dict(sorted(vars(args).items())).items():
        print(k)
    print()


def update_args(args):
    args.model_name = args.method + "_" + args.model_type + "_layers_" + str(args.unet_layers) + "_translatedMRI"

    #args.output_path = os.path.join(args.output_path, args.paired_dataset, args.extra_str, args.method, 'dice_fc_translated')
    args.output_path = os.path.join(args.output_path, args.paired_dataset, args.extra_str, args.method)
    args.model_save_path = os.path.join(args.model_save_path, args.paired_dataset, args.extra_str, args.method)
    return args


def main(args):
    seed_exps(args.seed)
    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    solver = Solver(args)
    solver.train()
    #solver.load_weights()
    #solver.viz_results()
    solver.test(train=False, cur_iter=args.total_iters+1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mri2pet')

    parser.add_argument('--dataset', default = 'mixed_1463', help='facades')
    parser.add_argument('--method', type=str, default='cnn_classifier', choices=['cnn', 'graph', 'cnn_classifier', 'graph_classifier', 'cnn_subv', 'graph_subv'])
    parser.add_argument('--seed', type=int, default=100, help='random seed to use.')
    
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--total_iters', type=int, default=200, help='training iters')
    parser.add_argument('--output_iter', type=int, default=50, help='training iters')
    parser.add_argument('--test_iter', type=int, default=1, help='training iters')
    parser.add_argument('--warmup', type=int, default=50, help='warmup iters')
    parser.add_argument('--resume_from_training', type=bool, default=False, help='Resume from saved checkpoint')
    
    parser.add_argument('--model_type', type=str, default='unet3d', help='the name of the model')
    #parser.add_argument('--n_gfilters', type=int, default=12, help='3d unet base filter number')
    #parser.add_argument('--n_dfilters', type=int, default=16, help='3d unet base filter number')
    parser.add_argument('--unet_layers', type=int, default=3, help='3d unet layers down/up')
    parser.add_argument('--cubic_len', type=int, default=256, help='smaller cubic len')
    parser.add_argument('--padding_len', type=int, default=24, help='padding')
    
    parser.add_argument('--data_path', type=str, default='/scratch/ktan24/documents/CMB_dataset/')
    #parser.add_argument('--data_path', type=str, default='/scratch/rpaul12/CMB_segmentation/datasets/')
    #parser.add_argument('--data_path', type=str, default='/data/amciilab/yaoxin/papaGAN_proejct/0629_datasets_copy/')
    parser.add_argument('--paired_dataset', type=str, default='ALL')
    #parser.add_argument('--paired_num', type = int, default = -1, help="-1 means all")
    #parser.add_argument('--unpaired_datasets', type=str, default='ADNI')
    parser.add_argument('--extra_str', type=str, default='mri2cmb', help='segmentation')

    
    parser.add_argument('--output_path', type=str, default='./seg_results/')
    parser.add_argument('--model_save_path', type=str, default='./saved_models/')
    parser.add_argument('--fold', type=int, default=1, help='selected fold')
    parser.add_argument('--n_workers', type=int, default=4, help='# workers')
    
    args = parser.parse_args()
    args = update_args(args)

    start_time = datetime.datetime.now()
    print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))

    print_args(args)
    main(args)

    end_time = datetime.datetime.now()
    print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
    duration = end_time - start_time
    print("Duration: " + str(duration))
