import os
import sys
import random

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils import utils
from models.model import AIDA
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from utils.EarlyStopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from utils.utils import (Dataset, calculate_prediction_slide, save_dict,
                         patch_slide_metrics, plot_roc_curve, cutoff_youdens_j, load_target_chunks)

class Train(object):
    def __init__(self, cfg):
        self.cfg     = cfg
        self.device  = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.writer  = SummaryWriter(log_dir=cfg['tensorboard_dir'])
        
        self.model   = AIDA(cfg['model']).to(self.device)


    def init(self, dataloader):
        if self.cfg['use_weighted_loss']:
            print(f'Using weight loss with weights of {dataloader.dataset.ratio}\n')
            weights = torch.FloatTensor(dataloader.dataset.ratio).to(self.device)
            self.criterion = torch.nn.CrossEntropyLoss(weight=weights)
            self.criterion_domain = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
            self.criterion_domain = torch.nn.CrossEntropyLoss()

        optimizer = getattr(torch.optim, self.cfg['optimizer'])
        self.optimizer = optimizer(self.model.parameters(), lr=self.cfg['lr'],
                                        weight_decay=self.cfg['wd'])
        self.early_stopping = EarlyStopping(patience=self.cfg['patience'],
                                            lr_patience=self.cfg['lr_patience'])
        if self.cfg['use_schedular']:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                        step_size=1, gamma=0.8)

    def optimize_parameters(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def forward(self, data, alpha):
        class_pred, domain_pred = self.model.forward(data, alpha)
        
        class_prob   = torch.softmax(class_pred, dim=1)
        domain_prob  = torch.softmax(domain_pred, dim=1)
        return class_pred, class_prob, domain_pred, domain_prob

    def train_one_epoch(self, train_dataloader, target_train_dataloader, epoch, all_epoch):
        """Train the model for a epoch
        Patch accuract will be shown at the end of the epoch

        Parameters
        ----------
        epoch: int
        train_dataloader : torch.utils.data.DataLoader
        """
        loss_ = 0
        gt_labels   = []
        pred_labels = []
        self.model.train()

        len_sourceloader = len(train_dataloader)

        i = 0
        prefix = f'Training Epoch {epoch}: '
        
        for data in tqdm(train_dataloader, desc=prefix,
                dynamic_ncols=True, leave=True, position=0):

            p = float(i + epoch * len_sourceloader) / all_epoch / len_sourceloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # Usually the size of target data is smaller than the source
            if i%len(target_train_dataloader)==0:
                data_target_iter = iter(target_train_dataloader)

            i += 1

            data, label, _, _ = data
            s_domain_label = torch.zeros(len(label)).long()

            data  = data.cuda() if torch.cuda.is_available() else data
            label = label.cuda() if torch.cuda.is_available() else label
            s_domain_label = s_domain_label.cuda() if torch.cuda.is_available() else s_domain_label

            
            sc_pred, sc_prob, sd_pred, sd_prob = self.forward(data, alpha)

            data_target = data_target_iter.next()
            t_data, t_label, _, _ = data_target
            t_domain_label = torch.ones(len(t_label)).long()

            t_data  = t_data.cuda() if torch.cuda.is_available() else t_data
            t_label = t_label.cuda() if torch.cuda.is_available() else t_label
            t_domain_label = t_domain_label.cuda() if torch.cuda.is_available() else t_domain_label

            _, _, td_pred, td_prob = self.forward(t_data, alpha)

            loss_s_label = self.criterion(sc_pred.type(torch.float), label.type(torch.long))
            loss_s_domain = self.criterion_domain(sd_pred.type(torch.float), s_domain_label.type(torch.long))
            loss_t_domain = self.criterion_domain(td_pred.type(torch.float), t_domain_label.type(torch.long))

            loss = loss_s_label + loss_s_domain + loss_t_domain

            self.optimize_parameters(loss)
            loss_ += loss.item() * label.shape[0]
            gt_labels   += label.cpu().numpy().tolist()
            pred_labels += torch.argmax(sc_prob, dim=1).cpu().numpy().tolist()

        train_acc  = accuracy_score(gt_labels, pred_labels)
        train_loss = loss_ / len(gt_labels)

        self.writer.add_scalar('train/train_loss', train_loss, global_step=epoch)
        self.writer.add_scalar('train/train_acc', train_acc, global_step=epoch)
        print(f"\nTraining (patch) accuracy is {train_acc:.4f} at epoch {epoch}.")

    def validate(self, dataloader, epoch=None, test=False):
        """Validate the model for a epoch by finding AUC of slides

        Parameters
        ----------
        epoch: int

        dataloader : torch.utils.data.DataLoader

        test : bool
            Whether it is in the test mode (true) or validation one (false)
        """
        loss_ = 0
        # slide level
        slide_idx_info = {} # {slide_id: {'patches': probs, 'gt_label': gt}}
        # patch level
        patch_info = {'gt_label': [], 'prediction': [],
                      'probability': np.array([]).reshape(0, self.cfg['num_classes'])}

        self.model.eval()
        txt = f'Validation Epoch {epoch}: ' if not test else 'Test : '
        with torch.no_grad():
            prefix = txt
            for data in tqdm(dataloader, desc=prefix,
                    dynamic_ncols=True, leave=True, position=0):

                data, label, slide_id, coords = data
                data  = data.cuda() if torch.cuda.is_available() else data
                label = label.cuda() if torch.cuda.is_available() else label
                predicted, prob, _, _ = self.forward(data, 0)
                if not test:
                    loss = self.criterion(predicted.type(torch.float), label.type(torch.long))
                    loss_ += loss.item() * label.shape[0]
                # patch level
                patch_info['gt_label']   += label.cpu().numpy().tolist()
                patch_info['prediction'] += torch.argmax(prob, dim=1).cpu().numpy().tolist()
                patch_info['probability'] = np.vstack((patch_info['probability'],
                                                       prob.cpu().numpy()))
                coords = [coords_.tolist() for coords_ in coords] # change tensor to list
                coords = np.array(coords).T.tolist() # change list of 2*n to n*2
                # slide level
                for (label_, prob_, slide_id_, coords_) in zip(label, prob, slide_id, coords):
                    if slide_id_ not in slide_idx_info:
                        slide_idx_info[slide_id_] = {}
                        slide_idx_info[slide_id_]['coords'] = []
                        slide_idx_info[slide_id_]['patches']  = np.array([]).reshape(0, self.cfg['num_classes'])
                    slide_idx_info[slide_id_]['gt_label'] = label_.item()
                    slide_idx_info[slide_id_]['coords'].append(coords_)
                    slide_idx_info[slide_id_]['patches'] = np.vstack((slide_idx_info[slide_id_]['patches'],
                                                                      prob_.cpu().numpy()))

        if not test:
            val_loss = loss_ / len(patch_info['gt_label'])
            self.early_stopping(val_loss)
        # slide level
        calculate_prediction_slide(slide_idx_info, self.cfg['num_classes'])
        # metris of both slide and patch level
        perf = patch_slide_metrics(slide_idx_info, patch_info, self.cfg['num_classes'])
        info = {'slide': slide_idx_info, 'performance': perf}
        # If in validation mode, shows the AUC and ACC
        if not test:
            for item in ['patch', 'slide']:
                val_auc  = perf[item]['overall_auc']
                val_acc  = perf[item]['overall_acc']
                val_bacc = perf[item]['balanced_acc']
                self.writer.add_scalar(f'validation/{item}_val_acc', val_acc, global_step=epoch)
                self.writer.add_scalar(f'validation/{item}_val_auc', val_auc, global_step=epoch)
                self.writer.add_scalar(f'validation/{item}_val_bacc', val_bacc, global_step=epoch)
                print(f"Validation ({item}) AUC is {val_auc:.4f}, ACC is {val_acc:.4f}, "
                      f"and BACC is {val_bacc:.4f}.")
        return info


    def train(self):
        print(f"\nStart Train training for {self.cfg['epochs']} epochs.")
        print(f"Training with {(self.device).upper()} device.\n")

        train_dataloader = Dataset(self.cfg['dataset'], state='train').run()
        valid_dataloader = Dataset(self.cfg['dataset'], state='validation').run()       

        target_train_dataloader = Dataset(self.cfg['dataset'], state='external_train').run()
        
        self.init(train_dataloader)

        best_valid_ = {}
        for method_ in self.cfg['save_method']:
            best_valid_[method_] = {}
            for criteria_ in self.cfg['criteria']:
                best_valid_[method_][criteria_] = -np.inf

        for epoch in range(self.cfg['epochs']):
            self.train_one_epoch(train_dataloader, target_train_dataloader, epoch, self.cfg['epochs'])
            info = self.validate(valid_dataloader, epoch)
            perf = info['performance']
            # check if in each method there are improvements based on both auc and acc
            # then save the model with this format model_{criteria_}_{method_}.pt
            for method_, criteria_value in best_valid_.items():
                for criteria, value in criteria_value.items():
                    if perf[method_][criteria] > value:
                        # save the model weights
                        best_valid_[method_][criteria] = perf[method_][criteria]
                        model_dict = self.model.state_dict() if not isinstance(self.model, nn.DataParallel) else \
                                     self.model.module.state_dict()
                        torch.save({'model': model_dict},
                                   os.path.join(self.cfg["checkpoints"], f"model_{method_}_{criteria}.pth"))
                        print(f"Saved model weights based on {method_} for {criteria} at epoch {epoch}.")
                        save_dict(info, f"{self.cfg['dict_dir']}/validation_{method_}_{criteria}.pkl")
            if self.early_stopping.early_stop:
                print("\nTraining has stopped because of early stopping!")
                break
            # in validation, model finds out the lr needs to be reduced.
            if self.cfg['use_schedular'] and self.early_stopping.reduce_lr:
                before_lr = self.scheduler.get_last_lr()
                self.scheduler.step()
                after_lr = self.scheduler.get_last_lr()
                print(f"\nLearning rate is decreased from {before_lr[0]} to {after_lr[0]}!")

        print("\nTraining has finished.")

    def test(self, use_external=False):
        """test the model by printing all the metrics for each saved model in both
        slide and patch level
        """
        if not use_external:
            print("\nStart Train testing.")
            test_dataloader = Dataset(self.cfg['dataset'], state='test').run()
        else:
            print(f"\nStart Train testing on external dataset ({self.cfg['external_test_name']}).")
            test_dataloader = Dataset(self.cfg['dataset'], state='external').run()

        output = '||Dataset||'
        for s in self.cfg['CategoryEnum']:
            output += f'{s.name} Accuracy||'
        output += 'Weighted Accuracy||Kappa||F1 Score||AUC||Average Accuracy||\n'

        output_patch = ''
        output_slide = ''

        conf_mtrs = {'slide_': {}, 'patch_': {}}

        for method_ in self.cfg['save_method']:
            conf_mtrs['slide_'][method_] = {}
            conf_mtrs['patch_'][method_] = {}
            for criteria_ in self.cfg['criteria']:

                path = os.path.join(self.cfg["checkpoints"], f"model_{method_}_{criteria_}.pth")
                state = torch.load(path, map_location=self.device)
                if isinstance(self.model, nn.DataParallel):
                    self.model.module.load_state_dict(state["model"], strict=True)
                else:
                    self.model.load_state_dict(state["model"], strict=True)

                name = f'{method_}_{criteria_}' if not use_external else f"{method_}_{criteria_}_external_{self.cfg['external_test_name']}"
                info = self.validate(test_dataloader, test=True)
                test_perf = info['performance']
                save_dict(info, f"{self.cfg['dict_dir']}/test_{name}.pkl")

                slide_metrics_ = test_perf['slide']
                patch_metrics_ = test_perf['patch']

                output_patch += f'|{method_}_{criteria_}|'
                output_slide += f'|{method_}_{criteria_}|'

                for i in range(self.cfg['num_classes']):
                    output_patch += f"{patch_metrics_['acc_per_subtype'][i]:.2f}%|"
                    output_slide += f"{slide_metrics_['acc_per_subtype'][i]:.2f}%|"
                output_patch += f"{patch_metrics_['overall_acc']*100:.2f}%|{patch_metrics_['overall_kappa']:.4f}|{patch_metrics_['overall_f1']:.4f}|{patch_metrics_['overall_auc']:.4f}|{patch_metrics_['acc_per_subtype'].mean():.2f}%|\n"
                output_slide += f"{slide_metrics_['overall_acc']*100:.2f}%|{slide_metrics_['overall_kappa']:.4f}|{slide_metrics_['overall_f1']:.4f}|{slide_metrics_['overall_auc']:.4f}|{slide_metrics_['acc_per_subtype'].mean():.2f}%|\n"

                conf_mtrs['slide_'][method_][criteria_] = slide_metrics_['conf_mat']
                conf_mtrs['patch_'][method_][criteria_] = patch_metrics_['conf_mat']

                roc_path = os.path.join(self.cfg["roc_dir"], f"{name}.png")
                plot_roc_curve(slide_metrics_['roc_curve'],
                            self.cfg['num_classes'], roc_path, test_dataloader)


        print('\n\n****** Slide Metric *******\n\n')
        print(output + output_slide)

        print(f'\nconfusion matrix')
        for method_ in self.cfg['save_method']:
            for criteria_ in self.cfg['criteria']:
                print()
                print(f'Method_{method_}_Criteria_{criteria_}')
                print(conf_mtrs['slide_'][method_][criteria_])

        print('\n\n****** Patch Metric *******\n\n')
        print(output + output_patch)

        print(f'\nconfusion matrix')
        for method_ in self.cfg['save_method']:
            for criteria_ in self.cfg['criteria']:
                print()
                print(f'Method_{method_}_Criteria_{criteria_}')
                print(conf_mtrs['patch_'][method_][criteria_])

        print("\nTesting has finished.")

    def run(self):
        if not self.cfg['only_test'] and not self.cfg['only_external_test']:
            self.train()
        if not self.cfg['only_external_test']:
            self.test()
        if self.cfg['test_external']:
            self.test(use_external=True)
