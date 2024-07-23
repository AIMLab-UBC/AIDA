import argparse
from torchvision import models
from utils.utils import (ParseKVToDictAction, handle_log_folder,
                         fix_random_seeds, subtype_kv, _cfg)
def parse_input():
    parser = argparse.ArgumentParser(description='SimpleTrain')
    parser.add_argument('--experiment_name', type=str,
                help="experiment name used to name log, model outputs.")
    parser.add_argument('--log_dir', type=str,
                help="directory to save the checkpoints and events files.")
    parser.add_argument('--chunk_file_location', type=str,
                help='path to JSON file contains patches address')
    parser.add_argument('--chunk_file_location_complete', type=str, default=None,
                help='path to JSON file contains all the possible patches address')
    parser.add_argument('--training_chunks', nargs="+", type=int, default=[0],
                help="space separated number IDs specifying chunks to use for training.")
    parser.add_argument('--validation_chunks', nargs="+", type=int, default=[1],
                help="space separated number IDs specifying chunks to use for validation.")
    parser.add_argument('--test_chunks', nargs="+", type=int, default=[2],
                help="space separated number IDs specifying chunks to use for validation.")
    parser.add_argument('--patch_pattern',
                help='patterns of the stored patches')
    parser.add_argument('--subtypes', nargs='+', type=subtype_kv, action=ParseKVToDictAction,
                help="space separated words describing subtype=groupping pairs for this study.")
    parser.add_argument('--num_classes', default=5, type=int,
                help='number of output classes')
    parser.add_argument('--epochs', default=5, type=int,
                help='number of total epochs to run')
    parser.add_argument('--num_patch_workers', default=4, type=int,
                help='number of data loading workers')
    parser.add_argument('--batch_size', default=256, type=int,
                help='batch size for trianing')
    parser.add_argument('--eval_batch_size', default=256, type=int,
                help='batch size for validation and testing phase')
    parser.add_argument('--resize', default=None, type=int,
                help='If set, resizing the image in augmentation.')
    parser.add_argument('--fft_enhancer', action='store_true',
                help='If set, AIDA will run, else ADA.')
    parser.add_argument('--lr', default=4e-5, type=float,
                help='initial learning rate')
    parser.add_argument('--wd', default=4e-5, type=float,
                help='weight decay')
    parser.add_argument('--seed', default=31, type=int,
                help='seed for initializing training')
    parser.add_argument('--patience', default=10, type=int,
                help='For early stopping')
    parser.add_argument('--lr_patience', default=5, type=int,
                help='For early stopping on learning rate')
    parser.add_argument('--save_method',
                choices=['patch', 'slide', 'All'], default='All',
                help='Method for saving trained model: 1.patch: based on '
                'validation patch accuracy 2. slide: based on validation slide '
                'accuracy 3. Both patch and slide accuracy')
    parser.add_argument('--optimizer',
                choices=['Adam', 'AdamW', 'SGD'], default='Adam',
                help='Optimizer for training the model: 1. Adam '
                '2. AdamW 3. SGD')
    parser.add_argument('--use_schedular', action='store_true',
                help='Using schedular for decreasig learning rate in a way that if '
                ' lr_patience has passed, it will be reduced by 0.8.')
    parser.add_argument('--not_use_weighted_loss', action='store_true',
                help='Not using weighted loss.')
    parser.add_argument('--criteria', type=str,
                choices=['overall_auc', 'overall_acc', 'balanced_acc', 'All'], default='All',
                help='Criteria for saving the best model: 1.overall_auc: use AUC same as original paper '
                'with highest probability 2.overall_acc: uses accuracy '
                '3.balanced_acc: balanced accuracy for imbalanced data '
                '4.All: uses all the possible criterias'
                'NOTE: For calculating AUC for multiclasses, OVO is used to mitigate '
                'the imbalanced classes.')
    parser.add_argument('--only_test', action='store_true',
                help='Only test not train.')
    parser.add_argument('--only_external_test', action='store_true',
                help='Only test on the external dataset.')
    parser.add_argument('--external_test_name', type=str,
                help="usefull when testing on multiple external datasets.")
    parser.add_argument('--external_chunk_file_location', type=str, default=None,
                help='path to JSON file contains external dataset')
    parser.add_argument('--external_chunks', nargs="+", type=int, default=[0,1,2],
                help="space separated number IDs specifying chunks to use for testing (default use all the slides).")

    args = parser.parse_args()
    return args

def parse_arguments():
    args = parse_input()
    cfg  = vars(args)
    handle_log_folder(cfg)
    fix_random_seeds(cfg["seed"])
    cfg = _cfg(cfg)
    print(cfg)
    return cfg

if __name__ == "__main__":
    print(parse_arguments())
