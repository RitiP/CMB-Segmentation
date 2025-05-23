import torch
import torch.nn as nn
import os
from torch import optim
import csv
from data_loader import paired_loader
#from data_loader_orig import paired_loader
from torchsummary import summary
from models.unet3d import *
from util import adjust_learning_rate, to_img, iou, dice_coeff, pixelwise_acc, dice_loss
from utils.evaluation_functions import PSNR, SSIM3D
import numpy as np
import torch.nn.functional as F
from focal_loss import *
import nibabel as nib
import pdb  
import time
from model import UNetWithClassifier
from losses import *



class Solver(object):
    def __init__(self, args):

        self.args = args
        #self.args.lr = 0.0002
        self.train_dataloader, self.val_dataloader, self.test_dataloader = paired_loader(self.args)
        # define the network here
        self.model = UNetWithClassifier(in_channels=1, out_channels=1, num_classes=2, final_sigmoid=False, f_maps=[16, 32, 64, 128], num_levels=4, is_segmentation=True).cuda()
        # self.model = UNetWithClassifier(in_channels=1, out_channels=2, num_classes=2, final_sigmoid=False, f_maps=[32, 64], num_levels=2, is_segmentation=True).cuda()
        # define the loss here, add focal loss later
        weights = [100.0]
        class_weights = torch.FloatTensor(weights).cuda()
        #self.seg_ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.ce_loss = nn.BCEWithLogitsLoss()
        self.seg_ce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(weight=class_weights, normalization='sigmoid') 
        #self.fc_loss = FocalLoss(alpha=class_weights, gamma=2)


        cur_time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + str(np.random.randint(low=0,high=100))
        print("Current time " + cur_time_str)

        self.save_info = True


    def train(self):

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), weight_decay=1e-8)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=10, min_lr=1e-10, verbose=True)

        best_dice = 0
        best_loss = float("inf")
        avg_train_dice = 0
        avg_train_dice_bg = 0
        avg_train_tp = 0
        avg_train_fp = 0
        avg_train_fn = 0
        self.cur_epoch = 0
        
        logger = {}
        logger['epochs'] = list()
        logger['loss'] = list()
        logger['dice'] = list()
        logger['dice_bg'] = list()
        logger['bce_loss'] = list()
        logger['seg_bce_loss'] = list()
        logger['dice_loss'] = list()

        test_logger = {}
        test_logger['epochs'] = list()
        test_logger['loss'] = list()
        test_logger['dice'] = list()
        test_logger['dice_bg'] = list()

        if self.args.resume_from_training:
            self.load_weights(resume=True)
            best_loss = self.best_loss
            best_dice = self.best_dice
            logger = self.train_logger
            test_logger = self.test_logger


        for i in range(self.cur_epoch, self.args.total_iters):
            self.model.train()  # Set the model to train mode
            total_dice = []
            total_dice_bg = []
            train_dice_loss, train_ce_loss, train_seg_ce_loss = 0, 0, 0
            total_loss = 0.0
            total_true_positives = 0
            total_false_positives = 0
            total_false_negatives = 0
            sample_metrics = []

            for fname, inputs, gt_mask, cmb_label in self.train_dataloader:
                inputs, gt_mask, cmb_label = inputs.cuda(), gt_mask.cuda(), cmb_label.cuda()
                inputs_shape = inputs.shape
                # reshape to (B*P,C,D,H,W), P - patches
                #inputs = inputs.view(inputs_shape[0]*inputs_shape[1], 1, inputs_shape[2], inputs_shape[3], inputs_shape[4])
                #gt_mask = gt_mask.view(inputs_shape[0]*inputs_shape[1], inputs_shape[2], inputs_shape[3], inputs_shape[4]) # 256, 1, 64, 64, 48
                #patch_labels = patch_labels.permute(1, 0)
                cmb_label = F.one_hot(cmb_label, num_classes=2)
                #pdb.set_trace()
                self.optimizer.zero_grad()

                pred_logits, pred_label = self.model(inputs)
                pred_mask = torch.sigmoid(pred_logits)
                pred_mask = (pred_mask > 0.1).long()
                #pred_mask = torch.argmax(pred_mask, dim=1)
                mask_one_hot = F.one_hot(gt_mask.long(), num_classes=1).permute(0, 4, 1, 2, 3).cuda() # 256, 2, 1, 64, 64, 48
                #patch_labels_oh = F.one_hot(patch_labels.long(), num_classes=2).squeeze().float().cuda()
                # loss calculation
                dice_loss = self.dice_loss(pred_logits, mask_one_hot.float())
                #pdb.set_trace()
                seg_ce_loss = self.seg_ce_loss(pred_logits, gt_mask.long())
                ce_loss = self.ce_loss(pred_label, cmb_label)
                batch_loss = dice_loss + seg_ce_loss + ce_loss

                # epoch loss
                train_dice_loss += dice_loss.item()
                train_ce_loss += ce_loss.item()
                train_seg_ce_loss += seg_ce_loss.item()
                total_loss += dice_loss.item() + seg_ce_loss.item() + ce_loss.item()
                
                # updates the parameters
                batch_loss.backward()
                self.optimizer.step()
                
                #outputs = outputs > 0.1
                metrics_dict = {}

                dice_bg, dice, dice_hash = dice_coeff(gt_mask, pred_mask, smooth=1e-6, num_classes=2)
                metrics_dict['fname'] = str(fname)
                metrics_dict['dice'] = dice
                metrics_dict['dice_bg'] = dice_bg

                # Convert outputs and labels to boolean for bitwise operations
                pred_mask = pred_mask.bool()
                gt_mask = gt_mask.bool()

                # Calculate intersection and union for overlap details
                true_positives = (pred_mask & gt_mask).float().sum()
                false_positives = (pred_mask & ~gt_mask).float().sum()
                false_negatives = (~pred_mask & gt_mask).float().sum()
                metrics_dict['TP'] = true_positives
                metrics_dict['FP'] = false_positives
                metrics_dict['FN'] = false_negatives

                sample_metrics.append(metrics_dict)
                del metrics_dict

                # Update totals
                total_dice.append(dice)
                total_dice_bg.append(dice_bg)
                total_true_positives += true_positives
                total_false_positives += false_positives
                total_false_negatives += false_negatives
            
            
            avg_train_dice = sum(total_dice)/len(total_dice)
            avg_train_dice_bg = sum(total_dice_bg)/len(total_dice_bg)
            avg_train_loss = total_loss/len(self.train_dataloader)
            avg_train_segce_loss = train_seg_ce_loss/len(self.train_dataloader)
            avg_train_ce_loss = train_ce_loss/len(self.train_dataloader)
            avg_train_dice_loss = train_dice_loss/len(self.train_dataloader)
            average_tp = total_true_positives/len(self.train_dataloader)
            average_fp = total_false_positives/len(self.train_dataloader)
            average_fn = total_false_negatives/len(self.train_dataloader)

            logger['epochs'].append(i)
            logger['loss'].append(avg_train_loss)
            logger['dice'].append(avg_train_dice)
            logger['dice_bg'].append(avg_train_dice_bg)
            logger['bce_loss'].append(avg_train_ce_loss)
            logger['seg_bce_loss'].append(avg_train_segce_loss)
            logger['dice_loss'].append(avg_train_dice_loss)

            print(f'Iteration:{i+1}/{self.args.total_iters}\tTrain Loss: {avg_train_loss}')
            print(f'Iteration:{i+1}/{self.args.total_iters}\tTrain Pixel-based BCE Loss: {avg_train_segce_loss}')
            print(f'Iteration:{i+1}/{self.args.total_iters}\tTrain BCE Loss: {avg_train_ce_loss}')
            print(f'Iteration:{i+1}/{self.args.total_iters}\tTrain Dice Loss: {avg_train_dice_loss}')
            print(f'Iteration:{i+1}/{self.args.total_iters}\tTrain DICE Coeff: {avg_train_dice}')
            print(f'Iteration:{i+1}/{self.args.total_iters}\tTrain DICE Coeff (with background): {avg_train_dice_bg}')
            print(f'Iteration:{i+1}/{self.args.total_iters}\tTrain True Positive: {average_tp}')
            print(f'Iteration:{i+1}/{self.args.total_iters}\tTrain False Positive: {average_fp}')
            print(f'Iteration:{i+1}/{self.args.total_iters}\tTrain False Negative: {average_fn}')

            if self.save_info:
                keys = sample_metrics[0].keys()

                with open(os.path.join(self.args.model_save_path, f'train_metrics_{i}.csv'), 'w', newline='\n') as output_file:
                    dict_writer = csv.DictWriter(output_file, keys)
                    dict_writer.writeheader()
                    dict_writer.writerows(sample_metrics)

            if (i+1) % self.args.test_iter == 0:
                print("Testing saved model...")
                val_dice, val_dice_bg, val_loss = self.val(train=True, cur_iter=i)
                #test_logger['epochs'].append(i)
                #test_logger['loss'].append(test_loss)
                #test_logger['dice'].append(test_dice)
                #test_logger['dice_bg'].append(test_dice_bg)
                
                if val_loss < best_loss:
                    torch.save(self.model.state_dict(), os.path.join(self.args.model_save_path, self.args.model_name+"_bestLoss.pt"))
                if val_dice_bg > best_dice:
                    torch.save(self.model.state_dict(), os.path.join(self.args.model_save_path, self.args.model_name+"_bestDICE.pt"))

                
                best_dice = max(val_dice_bg, best_dice)
                best_loss = min(val_loss, best_loss)
                print(f'Best test loss: {best_loss:.4f}')
                self.scheduler.step(val_loss)
                new_lr = self.optimizer.param_groups[0]['lr']
                print(f"Iteration:{i+1}\tlr: {new_lr: .8f}")
                
                print("Testing saved model...")
                test_dice, test_dice_bg, test_loss = self.test(train=True, cur_iter=i)
                test_logger['epochs'].append(i)
                test_logger['loss'].append(test_loss)
                test_logger['dice'].append(test_dice)
                test_logger['dice_bg'].append(test_dice_bg)
            
            torch.save({'epoch': i+1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'best_loss': best_loss,
                        'best_dice': best_dice,
                        'train_logger': logger,
                        'test_logger': test_logger}, os.path.join(self.args.model_save_path, self.args.model_name+"_latestEpoch.pt"))

        plot_and_save_training_metrics(logger, os.path.join(self.args.output_path, "train_logs.png"))
        plot_and_save_training_metrics(test_logger, os.path.join(self.args.output_path, "test_logs.png"))
    
    def val(self, train=False, cur_iter=0):
    
        self.model.eval()  # Set the model to evaluation mode
        total_dice = []
        total_dice_bg = []
        total_loss = 0.0
        dice_loss_test = 0
        ce_loss_test = 0
        seg_ce_loss_test = 0
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        count = 0
        sample_metrics = []

        with torch.no_grad():
            for fname, inputs, gt_mask, cmb_label in self.val_dataloader:
                inputs, gt_mask, cmb_label = inputs.cuda(), gt_mask.cuda(), cmb_label.cuda()
                cmb_label = F.one_hot(cmb_label, num_classes=2)
                inputs_shape = inputs.shape
                # inputs = inputs.view(inputs_shape[0]*inputs_shape[1], 1, inputs_shape[2], inputs_shape[3], inputs_shape[4])
                # gt_mask = gt_mask.view(inputs_shape[0]*inputs_shape[1], 1, inputs_shape[2], inputs_shape[3], inputs_shape[4])
                #inputs = inputs.view(inputs_shape[0]*inputs_shape[1], 1, inputs_shape[2], inputs_shape[3], inputs_shape[4])
                #gt_mask = gt_mask.view(inputs_shape[0]*inputs_shape[1], inputs_shape[2], inputs_shape[3], inputs_shape[4]) # 256, 1, 64, 64, 48                
                #patch_labels = patch_labels.permute(1, 0)
                
                pred_logits, pred_label = self.model(inputs)
                pred_mask = torch.sigmoid(pred_logits)
                pred_mask = (pred_mask > 0.1).long()
                # print(f"Max output value: {outputs.max().item()}, Min output value: {outputs.min().item()}")
                #pred_mask = torch.argmax(pred_mask, dim=1)
                mask_one_hot = F.one_hot(gt_mask.long(), num_classes=1).permute(0, 4, 1, 2, 3).cuda()
                #patch_labels_oh = F.one_hot(patch_labels.long(), num_classes=2).squeeze().float().cuda()
                # dice_loss = self.dice_loss(pred_mask, mask_one_hot.float())
                # seg_ce_loss = self.seg_ce_loss(pred_mask, gt_mask)
                dice_loss = self.dice_loss(pred_logits, mask_one_hot.float())
                seg_ce_loss = self.seg_ce_loss(pred_logits, gt_mask.long())
                ce_loss = self.ce_loss(pred_label, cmb_label)

                dice_loss_test += dice_loss.item()
                ce_loss_test += ce_loss.item()
                seg_ce_loss_test += seg_ce_loss.item()
                total_loss += dice_loss.item() + seg_ce_loss.item() + ce_loss.item()
                #outputs = outputs > 0.1
                metrics_dict = {}

                dice_bg, dice, dice_hash = dice_coeff(gt_mask, pred_mask, smooth=1e-6, num_classes=2)
                metrics_dict['fname'] = str(fname)
                metrics_dict['dice'] = dice
                metrics_dict['dice_bg'] = dice_bg

                # Convert outputs and labels to boolean for bitwise operations
                pred_mask = pred_mask.bool()
                gt_mask = gt_mask.bool()

                # Calculate intersection and union for overlap details
                true_positives = (pred_mask & gt_mask).float().sum()
                false_positives = (pred_mask & ~gt_mask).float().sum()
                false_negatives = (~pred_mask & gt_mask).float().sum()
                metrics_dict['TP'] = true_positives
                metrics_dict['FP'] = false_positives
                metrics_dict['FN'] = false_negatives

                sample_metrics.append(metrics_dict)
                del metrics_dict
                # Update totals
                total_dice.append(dice)
                total_dice_bg.append(dice_bg)
                total_true_positives += true_positives
                total_false_positives += false_positives
                total_false_negatives += false_negatives

                # # Calculate Dice coefficient for the current batch and accumulate
                # if(true_positives + false_positives + false_negatives>0):
                #     dice = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
                #     total_dice += dice.item()
                #     count += 1


        # Calculate average metrics
        #average_dice = 2 * total_true_positives / (2 * total_true_positives + total_false_positives + total_false_negatives)
        # print(f"Average Dice Coefficient: {average_dice:.4f}")
        average_dice = sum(total_dice)/len(total_dice)
        average_dice_bg = sum(total_dice_bg)/len(total_dice_bg)
        average_tp = total_true_positives/len(self.test_dataloader)
        average_fp = total_false_positives/len(self.test_dataloader)
        average_fn = total_false_negatives/len(self.test_dataloader)
        average_loss = total_loss/len(self.test_dataloader)
        avg_dice_loss = dice_loss_test/len(self.test_dataloader)
        avg_segce_loss = seg_ce_loss_test/len(self.test_dataloader)
        avg_ce_loss = ce_loss_test/len(self.test_dataloader)

        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest Loss: {average_loss}')
        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest Pixel-based BCE Loss: {avg_segce_loss}')
        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest BCE Loss: {avg_ce_loss}')
        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest Dice Loss: {avg_dice_loss}')
        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest DICE Coeff: {average_dice}')
        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest DICE Coeff (with background): {average_dice_bg}')
        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest True Positive: {average_tp}')
        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest False Positive: {average_fp}')
        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest False Negative: {average_fn}')


        if self.save_info:
            keys = sample_metrics[0].keys()

            with open(os.path.join(self.args.model_save_path, f'val_metrics_{cur_iter+1}.csv'), 'w', newline='\n') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(sample_metrics)

        print(f"{type} -> Average TP : {average_tp}, FP: {average_fp}, FN: {average_fn}")
        return average_dice, average_dice_bg, average_loss
    
    def test(self, train=False, cur_iter=0):
    
        self.model.eval()  # Set the model to evaluation mode
        total_dice = []
        total_dice_bg = []
        total_loss = 0.0
        dice_loss_test = 0
        ce_loss_test = 0
        seg_ce_loss_test = 0
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        count = 0
        sample_metrics = []

        with torch.no_grad():
            for fname, inputs, gt_mask, cmb_label in self.test_dataloader:
                inputs, gt_mask, cmb_label = inputs.cuda(), gt_mask.cuda(), cmb_label.cuda()
                cmb_label = F.one_hot(cmb_label, num_classes=2)
                inputs_shape = inputs.shape
                # inputs = inputs.view(inputs_shape[0]*inputs_shape[1], 1, inputs_shape[2], inputs_shape[3], inputs_shape[4])
                # gt_mask = gt_mask.view(inputs_shape[0]*inputs_shape[1], 1, inputs_shape[2], inputs_shape[3], inputs_shape[4])
                #inputs = inputs.view(inputs_shape[0]*inputs_shape[1], 1, inputs_shape[2], inputs_shape[3], inputs_shape[4])
                #gt_mask = gt_mask.view(inputs_shape[0]*inputs_shape[1], inputs_shape[2], inputs_shape[3], inputs_shape[4]) # 256, 1, 64, 64, 48                
                
                #patch_labels = patch_labels.permute(1, 0)
                
                pred_logits, pred_label = self.model(inputs)
                pred_mask = torch.sigmoid(pred_logits)
                pred_mask = (pred_mask > 0.1).long()
                # print(f"Max output value: {outputs.max().item()}, Min output value: {outputs.min().item()}")
                #pred_mask = torch.argmax(pred_mask, dim=1)
                mask_one_hot = F.one_hot(gt_mask.long(), num_classes=1).permute(0, 4, 1, 2, 3).cuda()
                #patch_labels_oh = F.one_hot(patch_labels.long(), num_classes=2).squeeze().float().cuda()
                # dice_loss = self.dice_loss(pred_mask, mask_one_hot.float())
                # seg_ce_loss = self.seg_ce_loss(pred_mask, gt_mask)
                dice_loss = self.dice_loss(pred_logits, mask_one_hot.float())
                seg_ce_loss = self.seg_ce_loss(pred_logits, gt_mask.long())
                ce_loss = self.ce_loss(pred_label, cmb_label)

                dice_loss_test += dice_loss.item()
                ce_loss_test += ce_loss.item()
                seg_ce_loss_test += seg_ce_loss.item()
                total_loss += dice_loss.item() + seg_ce_loss.item() + ce_loss.item()
                #outputs = outputs > 0.1
                metrics_dict = {}

                dice_bg, dice, dice_hash = dice_coeff(gt_mask, pred_mask, smooth=1e-6, num_classes=2)
                metrics_dict['fname'] = str(fname)
                metrics_dict['dice'] = dice
                metrics_dict['dice_bg'] = dice_bg

                # Convert outputs and labels to boolean for bitwise operations
                pred_mask = pred_mask.bool()
                gt_mask = gt_mask.bool()

                # Calculate intersection and union for overlap details
                true_positives = (pred_mask & gt_mask).float().sum()
                false_positives = (pred_mask & ~gt_mask).float().sum()
                false_negatives = (~pred_mask & gt_mask).float().sum()
                metrics_dict['TP'] = true_positives
                metrics_dict['FP'] = false_positives
                metrics_dict['FN'] = false_negatives

                sample_metrics.append(metrics_dict)
                del metrics_dict
                # Update totals
                total_dice.append(dice)
                total_dice_bg.append(dice_bg)
                total_true_positives += true_positives
                total_false_positives += false_positives
                total_false_negatives += false_negatives

                # # Calculate Dice coefficient for the current batch and accumulate
                # if(true_positives + false_positives + false_negatives>0):
                #     dice = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
                #     total_dice += dice.item()
                #     count += 1


        # Calculate average metrics
        #average_dice = 2 * total_true_positives / (2 * total_true_positives + total_false_positives + total_false_negatives)
        # print(f"Average Dice Coefficient: {average_dice:.4f}")
        average_dice = sum(total_dice)/len(total_dice)
        average_dice_bg = sum(total_dice_bg)/len(total_dice_bg)
        average_tp = total_true_positives/len(self.test_dataloader)
        average_fp = total_false_positives/len(self.test_dataloader)
        average_fn = total_false_negatives/len(self.test_dataloader)
        average_loss = total_loss/len(self.test_dataloader)
        avg_dice_loss = dice_loss_test/len(self.test_dataloader)
        avg_segce_loss = seg_ce_loss_test/len(self.test_dataloader)
        avg_ce_loss = ce_loss_test/len(self.test_dataloader)

        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest Loss: {average_loss}')
        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest Pixel-based BCE Loss: {avg_segce_loss}')
        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest BCE Loss: {avg_ce_loss}')
        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest Dice Loss: {avg_dice_loss}')
        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest DICE Coeff: {average_dice}')
        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest DICE Coeff (with background): {average_dice_bg}')
        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest True Positive: {average_tp}')
        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest False Positive: {average_fp}')
        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest False Negative: {average_fn}')


        if self.save_info:
            keys = sample_metrics[0].keys()

            with open(os.path.join(self.args.model_save_path, f'test_metrics_{cur_iter+1}.csv'), 'w', newline='\n') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(sample_metrics)

        print(f"{type} -> Average TP : {average_tp}, FP: {average_fp}, FN: {average_fn}")
        return average_dice, average_dice_bg, average_loss

    def load_weights(self, resume=False):
    
        if self.args.method == "cnn_classifier" or self.args.method == "cnn":
            if resume:
                checkpoint = torch.load(os.path.join(self.args.model_save_path, self.args.model_name+"_latestEpoch.pt"))
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.cur_epoch = checkpoint['epoch']
                self.best_loss = checkpoint['best_loss']
                self.best_dice = checkpoint['best_dice']
                self.train_logger = checkpoint['train_logger']
                self.test_logger = checkpoint['test_logger']
            else:
                #checkpoint = torch.load(os.path.join(self.args.model_save_path, self.args.model_name+".pt"))
                #pdb.set_trace()
                #self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.load_state_dict(torch.load(os.path.join(self.args.model_save_path, self.args.model_name+"_bestDICE.pt")))
        else:
            raise Exception("Not Implemented")


def plot_and_save_training_metrics(logger, filename='training_metrics.png'):
    epochs = logger['epochs']
    
    metrics_to_plot = [key for key in logger if key != 'epochs']
    num_metrics = len(metrics_to_plot)
    
    plt.figure(figsize=(15, 4 * num_metrics))

    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(num_metrics, 1, i + 1)
        plt.plot(epochs, logger[metric], marker='o', label=metric, linewidth=2)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.title(f'{metric} vs. Epochs', fontsize=14)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Training metrics plot saved as '{filename}'.")

