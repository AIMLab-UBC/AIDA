import os
import sys
import enum
import json
import pickle


import yaml
import torch
import random
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.sampler import BalancedBatchSampler
from utils.dataset import PatchDataset
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report, cohen_kappa_score, f1_score,
                             balanced_accuracy_score, roc_curve, auc,
                             silhouette_score)

def strip_extension(path):
    """Function to strip file extension

    Parameters
    ----------
    path : string
        Absoluate path to a slide

    Returns
    -------
    path : string
        Path to a file without file extension
    """
    p = Path(path)
    return str(p.with_suffix(''))

def create_patch_id(path, patch_pattern=None, rootpath=None):
    """Function to create patch ID either by
    1) patch_pattern to find the words to use for ID
    2) rootpath to clip the patch path from the left to form patch ID

    Parameters
    ----------
    path : string
        Absolute path to a patch

    patch_pattern : dict
        Dictionary describing the directory structure of the patch path. The words can be 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'.

    rootpath : str
        The root directory path containing patch to clip from patch path. Assumes patch contains rootpath.

    Returns
    -------
    patch_id : string
        Remove useless information before patch id for h5 file storage
    """
    if patch_pattern is not None:
        len_of_patch_id = -(len(patch_pattern) + 1)
        patch_id = strip_extension(path).split('/')[len_of_patch_id:]
        return '/'.join(patch_id)
    elif rootpath is not None:
        return strip_extension(path[len(rootpath):].lstrip('/'))
    else:
        return ValueError("Either patch_pattern or rootpath should be set.")

def get_label_by_patch_id(patch_id, patch_pattern, CategoryEnum, is_binary=False):
    """Get category label from patch id. The label can be either 'annotation' or 'subtype' based on is_binary flag.

    Parameters
    ----------
    patch_id : string
        Patch ID get label from

    patch_pattern : dict
        Dictionary describing the directory structure of the patch paths used to find the label word in the patch ID. The words can be 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'.

    CategoryEnum : enum.Enum
        Acts as the lookup table for category label

    is_binary : bool
        For binary classification, i.e., we will use BinaryEnum instead of SubtypeEnum

    Returns
    -------
    enum.Enum
        label from CategoryEnum
    """
    label = patch_id.split('/')[patch_pattern['annotation' if is_binary else 'subtype']]
    return CategoryEnum[label if is_binary else label.upper()]

def get_slide_by_patch_id(patch_id, patch_pattern):
    """Function to obtain slide id from patch id

    Parameters
    ----------
    patch_id : str

    patch_pattern : dict
        Dictionary describing the directory structure of the patch paths used to find the slide word in the patch ID. The words can be 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'.

    Returns
    -------
    slide_id : str
        Slide id extracted from `patch_id`
    """
    slide_id = patch_id.split('/')[patch_pattern['slide']]
    return slide_id

def load_target_chunks(chunk_file_location, chunk_ids):
    """Load patch paths from specified chunks in chunk file

    Parameters
    ----------
    chunks : list of int
        The IDs of chunks to retrieve patch paths from

    Returns
    -------
    list of list
        [Patch paths, slide_ID] from the chunks
    """
    patch_paths = []
    with open(chunk_file_location) as f:
        data = json.load(f)
        chunks = data['chunks']
        for chunk in data['chunks']:
            if chunk['id'] in chunk_ids:
                patch_paths.extend([path for path in chunk['imgs']])
    if len(patch_paths) == 0:
        raise ValueError(
                f"chunks {tuple(chunk_ids)} not found in {chunk_file_location}")
    
    # print(patch_paths)
    return patch_paths


def load_chunks(chunk_file_location, chunk_ids, patch_pattern):
    """Load patch paths from specified chunks in chunk file

    Parameters
    ----------
    chunks : list of int
        The IDs of chunks to retrieve patch paths from

    Returns
    -------
    list of list
        [Patch paths, slide_ID] from the chunks
    """
    patch_paths = []
    with open(chunk_file_location) as f:
        data = json.load(f)
        chunks = data['chunks']
        for chunk in data['chunks']:
            if chunk['id'] in chunk_ids:
                patch_paths.extend([[path, get_slide_by_patch_id(create_patch_id(path,
                                                    patch_pattern), patch_pattern)] for path in chunk['imgs']])
    if len(patch_paths) == 0:
        raise ValueError(
                f"chunks {tuple(chunk_ids)} not found in {chunk_file_location}")
    return patch_paths

def extract_label_from_patch(CategoryEnum, patch_pattern, patch_path):
    """Get the label value according to CategoryEnum from the patch path

    Parameters
    ----------
    patch_path : str

    Returns
    -------
    int
        The label id for the patch
    """
    '''
    Returns the CategoryEnum
    '''
    patch_path = patch_path[0]
    patch_id = create_patch_id(patch_path, patch_pattern)
    label = get_label_by_patch_id(patch_id, patch_pattern,
            CategoryEnum, is_binary=False)
    return label.value

def extract_labels(CategoryEnum, patch_pattern, patch_paths):
    return [extract_label_from_patch(CategoryEnum, patch_pattern, path) for path in patch_paths]

def find_slide_idx(patch_paths):
    """Find which patches corresponds to specific slide id

    Parameters
    ----------
    patch_paths : list of str

    Returns
    -------
    dict : dict {slide_id: [list of idx ...]}
    """
    dict = {}
    for idx, path_slide in enumerate(patch_paths):
        # path = path_slide[0]
        slide_id = path_slide[1]
        if slide_id not in dict: dict[slide_id] = []
        dict[slide_id].append(idx)
    return dict

def create_data_set(cfg, chunk_id, state, training_set=True):
    """Create dataset

    Parameters
    ----------
    cfg : dict
        config file

    chunk_id : int

    state: str

    training_set: bool
        whether activate augmentation (in traning mode) or not

    Returns
    -------
    patch_dataset : Dataset
    """
    patch_pattern = {k: i for i, k in enumerate(cfg["patch_pattern"].split('/'))}
    # load the complete dataset for validation and test
    if state == 'train':
        chunk_file = cfg["chunk_file_location"]
    elif state == 'validation' or state == 'test':
        chunk_file = cfg["chunk_file_location"] if cfg["chunk_file_location_complete"] is None \
            else cfg["chunk_file_location_complete"]
    elif state == 'external' or state == 'external_train':
        chunk_file = cfg["external_chunk_file_location"]
    patch_paths  = load_chunks(chunk_file, chunk_id, patch_pattern)
    CategoryEnum = enum.Enum('SubtypeEnum', cfg["subtypes"])
    slide_idx = find_slide_idx(patch_paths)
    labels = extract_labels(CategoryEnum, patch_pattern, patch_paths)

    target_paths = load_target_chunks(cfg['external_chunk_file_location'],
                                      cfg['external_chunks'])
    patch_dataset = PatchDataset(patch_paths, labels, cfg["CategoryEnum"], state,
                                 cfg["resize"], cfg['fft_enhancer'], target_paths, training_set=training_set)
    return patch_dataset, labels

class Dataset(object):
    def __init__(self, cfg, state='train'):
        self.cfg = cfg
        self.state = state
        chunk, training_set = self.handle_chunk()
        self.patch_dataset, self.labels = create_data_set(cfg, chunk, state, training_set)

    def handle_chunk(self):
        if self.state == 'train':
            chunk = self.cfg["training_chunks"]
            training_set = True
        elif self.state == 'validation':
            chunk = self.cfg["validation_chunks"]
            training_set = False
        elif self.state == 'test':
            chunk = self.cfg["test_chunks"]
            training_set = False
        elif self.state == 'external' or self.state == 'external_train':
            chunk = self.cfg["external_chunks"]
            training_set = False
        else:
            raise ValueError(f'{state} should be either train, validation, test or external test!')
        return chunk, training_set

    def run(self):
        batch_size = self.cfg["batch_size"] if self.state=='train' else \
                     self.cfg["eval_batch_size"]

        return DataLoader(self.patch_dataset, batch_size=batch_size,
                      shuffle=True, pin_memory=True,
                      num_workers=self.cfg["num_patch_workers"])

def calculate_prediction_slide(slide_idx_info, num_class):
    """Calculate the predicted label and probability based for three possible method

    Parameters
    ----------
    slide_idx_prob   : dict {slide_id: {'patches': probs, 'gt_label': gt}}
    thresh : dict
    """
    for slide, info in slide_idx_info.items():
        patch_labels = np.argmax(info['patches'], axis=1)
        # majority voting
        counts = np.bincount(patch_labels, minlength=num_class)
        slide_idx_info[slide]['prediction']  = np.argmax(counts)
        slide_idx_info[slide]['probability'] = counts / sum(counts)


def metrics_(gt_labels, pred_labels, pred_probs, num_classes):
    pred_probs = np.asarray(pred_probs)
    if num_classes > 2:
        overall_auc = roc_auc_score(gt_labels, pred_probs, multi_class='ovr', average='macro')
    else:
        overall_auc = roc_auc_score(gt_labels, pred_probs[:, 1], average='macro')
    overall_acc     = accuracy_score(gt_labels, pred_labels)
    balanced_acc    = balanced_accuracy_score(gt_labels, pred_labels)
    overall_kappa   = cohen_kappa_score(gt_labels, pred_labels)
    overall_f1      = f1_score(gt_labels, pred_labels, average='macro')
    conf_mat        = confusion_matrix(gt_labels, pred_labels).T
    acc_per_subtype = conf_mat.diagonal() / conf_mat.sum(axis=0) * 100
    acc_per_subtype[np.isinf(acc_per_subtype)] = 0.00

    # roc curve for classes
    ovr_roc_curve = {'fpr': {}, 'tpr': {}, 'thresh': {}}
    for num_ in range(num_classes):
        ovr_roc_curve['fpr'][num_], ovr_roc_curve['tpr'][num_], ovr_roc_curve['thresh'][num_] = roc_curve(gt_labels,
                                                pred_probs[:, num_], pos_label=num_)

    return {'overall_auc': overall_auc, 'overall_acc': overall_acc, 'overall_kappa': overall_kappa,
            'overall_f1': overall_f1, 'conf_mat': conf_mat, 'acc_per_subtype': acc_per_subtype,
            'balanced_acc': balanced_acc, 'roc_curve': ovr_roc_curve}

def patch_slide_metrics(slide_idx_info, patch_info, num_classes):
    """Calculate all metrics for given dataloader
    """
    metrics = {}
    metrics['patch'] = metrics_(patch_info['gt_label'], patch_info['prediction'],
                                patch_info['probability'], num_classes)

    slide_gt_label = []
    slide_pred     = []
    slide_prob     = np.array([]).reshape(0, num_classes)
    for info in slide_idx_info.values():
        slide_gt_label.append(info['gt_label'])
        slide_pred.append(info['prediction'])
        slide_prob = np.vstack((slide_prob, info['probability']))

    metrics['slide'] = metrics_(slide_gt_label, slide_pred,
                                slide_prob, num_classes)
    return metrics

def cutoff_youdens_j(fpr, tpr, thresh):
    opt_idx    = np.argmax(tpr - fpr)
    opt_thresh = thresh[opt_idx]
    return opt_thresh

def plot_roc_curve(ovr_roc_curve, num_classes, path, dataloader):
    plt.figure()
    if num_classes > 2:
        for i in range(num_classes):
            plt.plot(ovr_roc_curve['fpr'][i], ovr_roc_curve['tpr'][i],
            label=f"class {dataloader.dataset.classes[i]} vs rest (area = "
                  f"{auc(ovr_roc_curve['fpr'][i], ovr_roc_curve['tpr'][i]):0.2f})")
    else:
        i = 1
        plt.plot(ovr_roc_curve['fpr'][i], ovr_roc_curve['tpr'][i],
        label=f"{dataloader.dataset.classes[0]} vs {dataloader.dataset.classes[1]} (area = "
              f"{auc(ovr_roc_curve['fpr'][i], ovr_roc_curve['tpr'][i]):0.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(path)

def save_dict(dict, path):
    with open(path, 'wb') as f:
        pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)

######################################
#              Parser                #
######################################
class ParseKVToDictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, type=None, **kwargs):
        if nargs != '+':
            raise argparse.ArgumentTypeError(f"ParseKVToDictAction can only be used for arguments with nargs='+' but instead we have nargs={nargs}")
        super(ParseKVToDictAction, self).__init__(option_strings, dest,
                nargs=nargs, type=type, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, option_string.lstrip('-'), make_dict(values))

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    Copy from: https://github.com/facebookresearch/dino/blob/main/utils.py
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def handle_log_folder(cfg):
    cfg["log_dir"]     = os.path.join(cfg["log_dir"], f"{cfg['experiment_name']}")
    cfg["roc_dir"]     = os.path.join(cfg["log_dir"], 'roc_curves')
    cfg["dict_dir"]    = os.path.join(cfg["log_dir"], 'information')
    cfg["checkpoints"] = os.path.join(cfg["log_dir"], 'checkpoints')
    cfg["tensorboard_dir"] = os.path.join(cfg["log_dir"], 'tensorboard')

    os.makedirs(cfg["log_dir"], exist_ok=True)
    os.makedirs(cfg["roc_dir"], exist_ok=True)
    os.makedirs(cfg["dict_dir"], exist_ok=True)
    os.makedirs(cfg["checkpoints"], exist_ok=True)
    os.makedirs(cfg["tensorboard_dir"], exist_ok=True)

def make_dict(ll):
    return {k: v for (k, v) in ll}

def subtype_kv(kv):
    """Used to identify and convert key=value arguments into a tuple (key.upper(), int(value)).
    For example: MMRd=0 becomes (MMRD, int(0))
    This is to be passed as the type when calling argparse.ArgumentParser.add_argument()

    Parameters
    ----------
    kv: str
        a key=value argument

    Returns
    -------
    tuple
        (key.upper(), int(value)) from key=value
    """
    try:
        k, v = kv.split("=")
    except:
        raise argparse.ArgumentTypeError(f"value {kv} is not separated by one '='")
    k = k.upper()
    try:
        v = int(v)
    except:
        raise argparse.ArgumentTypeError(f"right side of {kv} should be int")
    return (k, v)

def _cfg(cfg):
    cfg_ = {}

    # model
    cfg_['model'] =  {'num_classes': cfg['num_classes']}

    enum_ = enum.Enum('SubtypeEnum', cfg["subtypes"])

    # dataset
    cfg_['dataset'] =  {'training_chunks': cfg['training_chunks'],
                        'validation_chunks': cfg['validation_chunks'],
                        'test_chunks': cfg['test_chunks'],
                        'batch_size': cfg['batch_size'],
                        'eval_batch_size': cfg['eval_batch_size'],
                        'num_patch_workers': cfg['num_patch_workers'],
                        'subtypes': cfg['subtypes'],
                        'patch_pattern': cfg['patch_pattern'],
                        'chunk_file_location': cfg['chunk_file_location'],
                        'chunk_file_location_complete': cfg['chunk_file_location_complete'],
                        'external_chunk_file_location': cfg['external_chunk_file_location'],
                        'external_chunks': cfg['external_chunks'],
                        'CategoryEnum': enum_,
                        'resize': cfg['resize'],
                        'fft_enhancer': cfg['fft_enhancer']}

    criteria_ = ['overall_auc', 'overall_acc', 'balanced_acc'] if cfg['criteria'] == 'All' else [cfg['criteria']]
    save_method_ = ['patch', 'slide'] if cfg['save_method'] == 'All' else [cfg['save_method']]

    test_external = True if cfg['external_chunk_file_location'] is not None else False

    cfg_['lr']                   = cfg['lr']
    cfg_['wd']                   = cfg['wd']
    cfg_['criteria']             = criteria_
    cfg_['epochs']               = cfg['epochs']
    cfg_['log_dir']              = cfg['log_dir']
    cfg_['roc_dir']              = cfg['roc_dir']
    cfg_['dict_dir']             = cfg['dict_dir']
    cfg_['patience']             = cfg['patience']
    cfg_['optimizer']            = cfg['optimizer']
    cfg_['only_test']            = cfg['only_test']
    cfg_['lr_patience']          = cfg['lr_patience']
    cfg_['num_classes']          = cfg['num_classes']
    cfg_['checkpoints']          = cfg['checkpoints']
    cfg_['save_method']          = save_method_
    cfg_['CategoryEnum']         = enum_
    cfg_['use_schedular']        = cfg['use_schedular']
    cfg_['test_external']        = test_external
    cfg_['tensorboard_dir']      = cfg['tensorboard_dir']
    cfg_['use_weighted_loss']    = not cfg['not_use_weighted_loss']
    cfg_['only_external_test']   = cfg['only_external_test']
    cfg_['external_test_name']   = cfg['external_test_name']

    assert len([None for k in cfg_['CategoryEnum']])==cfg_['num_classes'], \
        f"Number of classes does not match with subtypes!"

    return cfg_
