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
from model import UNetWithClassifier, UNet3D
from losses import *
import itertools

def sample_positives(gt_mask):
    """
    gt_mask: Tensor [B, D, H, W], binary {0,1}
    Returns: LongTensor of shape [N_pos,4] with (b, d, h, w) coords
    """
    return torch.nonzero(gt_mask==1, as_tuple=False)

def sample_negatives(gt_mask, avoid_coords, num_neg):
    """
    gt_mask:       [B, D, H, W] binary
    avoid_coords:  LongTensor [N_pos,4] coords to avoid
    num_neg:       int, number of negatives desired
    Returns: LongTensor [num_neg,4]
    """
    #B,D,H,W = gt_mask.shape
    B,C,D,H,W = gt_mask.shape
    # Gather all background coords
    all_bg = []
    for b in range(B):
        pos = avoid_coords[avoid_coords[:,0]==b][:,1:]
        # create a mask to exclude pos coords
        mask = torch.ones((C,D,H,W), dtype=torch.bool, device=gt_mask.device)
        if pos.numel()>0:
            mask[pos[:,0], pos[:,1], pos[:,2], pos[:,3]] = False
        coords = torch.nonzero((gt_mask[b]==0) & mask, as_tuple=False)
        bcol = torch.full((coords.size(0),1), b, device=coords.device, dtype=torch.long)
        all_bg.append(torch.cat([bcol, coords], dim=1))
    all_bg = torch.cat(all_bg, dim=0)
    if all_bg.size(0) <= num_neg:
        return all_bg
    idx = torch.randperm(all_bg.size(0), device=all_bg.device)[:num_neg]
    return all_bg[idx]

class PatchContrastiveLoss(nn.Module):
    def __init__(self, tau=0.07):
        super().__init__()
        self.tau = tau

    def forward(self, Z, labels):
        # Z: [M, d], labels: [M] in {0,1}
        sim = (Z @ Z.T) / self.tau            # [M,M]
        mask = labels.unsqueeze(1).eq(labels.unsqueeze(0)).float()
        mask.fill_diagonal_(0)                # exclude self
        exp_sim = torch.exp(sim) * (1 - torch.eye(len(Z), device=Z.device))
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        return -mean_log_prob_pos.mean()


class Solver(object):
    def __init__(self, args):

        self.args = args
        #self.args.lr = 0.0002
        self.train_dataloader, self.val_dataloader, self.test_dataloader = paired_loader(self.args)
        # define the network here
        self.model = UNet3D(in_channels=1, out_channels=1, num_classes=2, final_sigmoid=False, f_maps=[16, 32, 64, 128], num_levels=4, is_segmentation=True).cuda()
        self.proj_head = nn.Sequential(
                nn.Linear(17, 32), nn.ReLU(inplace=True),
                nn.Linear(32, 16)).cuda()
        # self.model = UNetWithClassifier(in_channels=1, out_channels=2, num_classes=2, final_sigmoid=False, f_maps=[32, 64], num_levels=2, is_segmentation=True).cuda()
        # define the loss here, add focal loss later
        self.con_batch = 64
        weights = [100.0]
        class_weights = torch.FloatTensor(weights).cuda()
        #self.seg_ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        #self.ce_loss = nn.BCEWithLogitsLoss()
        self.seg_ce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(weight=class_weights, normalization='sigmoid') 
        self.contrastive_loss = PatchContrastiveLoss(tau=0.1)
        #self.fc_loss = FocalLoss(alpha=class_weights, gamma=2)


        cur_time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + str(np.random.randint(low=0,high=100))
        print("Current time " + cur_time_str)

        self.save_info = True


    def train(self):

        self.optimizer = optim.Adam(itertools.chain(self.model.parameters(), self.proj_head.parameters()), lr=self.args.lr, betas=(0.9, 0.999), weight_decay=1e-8)
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
            train_dice_loss, train_ce_loss, train_seg_ce_loss, train_con_loss = 0, 0, 0, 0
            total_loss = 0.0
            total_true_positives = 0
            total_false_positives = 0
            total_false_negatives = 0
            sample_metrics = []

            for fname, inputs, gt_mask, cmb_label in self.train_dataloader:
                inputs, gt_mask, cmb_label = inputs.cuda(), gt_mask.cuda(), cmb_label.cuda()
                inputs_shape = inputs.shape
                #print(fname)
                # reshape to (B*P,C,D,H,W), P - patches
                #inputs = inputs.view(inputs_shape[0]*inputs_shape[1], 1, inputs_shape[2], inputs_shape[3], inputs_shape[4])
                #gt_mask = gt_mask.view(inputs_shape[0]*inputs_shape[1], inputs_shape[2], inputs_shape[3], inputs_shape[4]) # 256, 1, 64, 64, 48
                #patch_labels = patch_labels.permute(1, 0)
                cmb_label = F.one_hot(cmb_label, num_classes=2)
                #pdb.set_trace()
                self.optimizer.zero_grad()

                pred_logits, decoder_feats = self.model(inputs)
                pred_mask = torch.sigmoid(pred_logits)
                pred_mask = (pred_mask > 0.1).long()
                #pred_mask = torch.argmax(pred_mask, dim=1)
                #mask_one_hot = F.one_hot(gt_mask.long(), num_classes=1).permute(0, 4, 1, 2, 3).cuda() # 256, 2, 1, 64, 64, 48
                mask_one_hot = gt_mask
                #patch_labels_oh = F.one_hot(patch_labels.long(), num_classes=2).squeeze().float().cuda()
                # loss calculation
                dice_loss = self.dice_loss(pred_logits, mask_one_hot.float())
                #pdb.set_trace()
                seg_ce_loss = self.seg_ce_loss(pred_logits, gt_mask.float())
                #ce_loss = self.ce_loss(pred_label, cmb_label.float())
                

                pos_coords = sample_positives(gt_mask)
                neg_coords = sample_negatives(gt_mask, pos_coords, self.con_batch)
                # if CMB volume, start contrastive loss
                if pos_coords.size(0) > 0:
                    all_coords = torch.cat([pos_coords, neg_coords], dim=0)
                    labels = torch.cat([torch.ones(len(pos_coords)), torch.zeros(len(neg_coords))], 0).long().to(inputs.device)

                    embs = []
                    for (b,c,d,h,w) in all_coords:
                        f_patch = decoder_feats[b, :, d-10:d+10, h-10:h+10, w-10:w+10]  # [16,20,20,20]
                        l_patch = pred_logits[b, :, d-10:d+10, h-10:h+10, w-10:w+10]      # [1,20,20,20]
                        v_feat = f_patch.contiguous().view(16, -1).mean(dim=1)
                        l_feat = l_patch.contiguous().view(1, -1).mean(dim=1)
                        embs.append(torch.cat([v_feat, l_feat], dim=0))
                    E = torch.stack(embs, dim=0)  # [2P,17]
                
                    # project & normalize
                    Z = F.normalize(self.proj_head(E), dim=1)  # define proj_head in __init__
                
                    # contrastive loss
                    loss_con = self.contrastive_loss(Z, labels)
                else:
                    loss_con = torch.tensor(0.).cuda()

                batch_loss = dice_loss + seg_ce_loss + loss_con # + ce_loss

                # epoch loss
                train_dice_loss += dice_loss.item()
                #train_ce_loss += ce_loss.item()
                train_seg_ce_loss += seg_ce_loss.item()
                train_con_loss += loss_con.item()
                total_loss += dice_loss.item() + seg_ce_loss.item() + loss_con.item() # + ce_loss.item()

                
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
            avg_train_con_loss = train_con_loss/len(self.train_dataloader)
            #avg_train_ce_loss = train_ce_loss/len(self.train_dataloader)
            avg_train_dice_loss = train_dice_loss/len(self.train_dataloader)
            average_tp = total_true_positives/len(self.train_dataloader)
            average_fp = total_false_positives/len(self.train_dataloader)
            average_fn = total_false_negatives/len(self.train_dataloader)

            logger['epochs'].append(i)
            logger['loss'].append(avg_train_loss)
            logger['dice'].append(avg_train_dice)
            logger['dice_bg'].append(avg_train_dice_bg)
            #logger['bce_loss'].append(avg_train_ce_loss)
            logger['seg_bce_loss'].append(avg_train_segce_loss)
            logger['dice_loss'].append(avg_train_dice_loss)

            print(f'Iteration:{i+1}/{self.args.total_iters}\tTrain Loss: {avg_train_loss}')
            print(f'Iteration:{i+1}/{self.args.total_iters}\tTrain Pixel-based BCE Loss: {avg_train_segce_loss}')
            #print(f'Iteration:{i+1}/{self.args.total_iters}\tTrain BCE Loss: {avg_train_ce_loss}')
            print(f'Iteration:{i+1}/{self.args.total_iters}\tTrain Dice Loss: {avg_train_dice_loss}')
            print(f'Iteration:{i+1}/{self.args.total_iters}\tTrain Contrastive Loss: {avg_train_con_loss}')
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
        self.proj_head.eval()
        total_dice = []
        total_dice_bg = []
        total_loss = 0.0
        dice_loss_test = 0
        ce_loss_test = 0
        seg_ce_loss_test = 0
        con_loss_test = 0
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
                
                pred_logits, decoder_feats = self.model(inputs)
                pred_mask = torch.sigmoid(pred_logits)
                pred_mask = (pred_mask > 0.1).long()
                # print(f"Max output value: {outputs.max().item()}, Min output value: {outputs.min().item()}")
                #pred_mask = torch.argmax(pred_mask, dim=1)
                #mask_one_hot = F.one_hot(gt_mask.long(), num_classes=1).permute(0, 4, 1, 2, 3).cuda()
                mask_one_hot = gt_mask
                #patch_labels_oh = F.one_hot(patch_labels.long(), num_classes=2).squeeze().float().cuda()
                # dice_loss = self.dice_loss(pred_mask, mask_one_hot.float())
                # seg_ce_loss = self.seg_ce_loss(pred_mask, gt_mask)
                dice_loss = self.dice_loss(pred_logits, mask_one_hot.float())
                seg_ce_loss = self.seg_ce_loss(pred_logits, gt_mask.float())
                #ce_loss = self.ce_loss(pred_label, cmb_label.float())

                dice_loss_test += dice_loss.item()
                #ce_loss_test += ce_loss.item()
                seg_ce_loss_test += seg_ce_loss.item()

                pos_coords = sample_positives(gt_mask)
                neg_coords = sample_negatives(gt_mask, pos_coords, self.con_batch)
                # if CMB volume, start contrastive loss
                if pos_coords.size(0) > 0:
                    all_coords = torch.cat([pos_coords, neg_coords], dim=0)
                    labels = torch.cat([torch.ones(len(pos_coords)), torch.zeros(len(neg_coords))], 0).long().to(inputs.device)

                    embs = []
                    for (b,d,h,w) in all_coords:
                        f_patch = decoder_feats[b, :, d-10:d+10, h-10:h+10, w-10:w+10]  # [16,20,20,20]
                        l_patch = pred_logits[b, :, d-10:d+10, h-10:h+10, w-10:w+10]      # [1,20,20,20]
                        v_feat = f_patch.view(16, -1).mean(dim=1)
                        l_feat = l_patch.view(1, -1).mean(dim=1)
                        embs.append(torch.cat([v_feat, l_feat], dim=0))
                    E = torch.stack(embs, dim=0)  # [2P,17]
                
                    # project & normalize
                    Z = F.normalize(self.proj_head(E), dim=1)  # define proj_head in __init__
                
                    # contrastive loss
                    loss_con = self.contrastive_loss(Z, labels)
                else:
                    loss_con = torch.tensor(0.).cuda()

                con_loss_test += loss_con.item()
                total_loss += dice_loss.item() + seg_ce_loss.item() + loss_con.item() # + ce_loss.item()
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
        avg_con_loss = con_loss_test/len(self.test_dataloader)
        #avg_ce_loss = ce_loss_test/len(self.test_dataloader)

        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest Loss: {average_loss}')
        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest Pixel-based BCE Loss: {avg_segce_loss}')
        #print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest BCE Loss: {avg_ce_loss}')
        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest Dice Loss: {avg_dice_loss}')
        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest Contrastive Loss: {avg_con_loss}')
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
        con_loss_test = 0
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
                
                pred_logits, decoder_feats = self.model(inputs)
                pred_mask = torch.sigmoid(pred_logits)
                pred_mask = (pred_mask > 0.1).long()
                # print(f"Max output value: {outputs.max().item()}, Min output value: {outputs.min().item()}")
                #pred_mask = torch.argmax(pred_mask, dim=1)
                #mask_one_hot = F.one_hot(gt_mask.long(), num_classes=1).permute(0, 4, 1, 2, 3).cuda()
                mask_one_hot = gt_mask
                #patch_labels_oh = F.one_hot(patch_labels.long(), num_classes=2).squeeze().float().cuda()
                # dice_loss = self.dice_loss(pred_mask, mask_one_hot.float())
                # seg_ce_loss = self.seg_ce_loss(pred_mask, gt_mask)
                dice_loss = self.dice_loss(pred_logits, mask_one_hot.float())
                seg_ce_loss = self.seg_ce_loss(pred_logits, gt_mask.float())
                #ce_loss = self.ce_loss(pred_label, cmb_label.float())

                dice_loss_test += dice_loss.item()
                #ce_loss_test += ce_loss.item()
                seg_ce_loss_test += seg_ce_loss.item()

                pos_coords = sample_positives(gt_mask)
                neg_coords = sample_negatives(gt_mask, pos_coords, self.con_batch)
                # if CMB volume, start contrastive loss
                if pos_coords.size(0) > 0:
                    all_coords = torch.cat([pos_coords, neg_coords], dim=0)
                    labels = torch.cat([torch.ones(len(pos_coords)), torch.zeros(len(neg_coords))], 0).long().to(inputs.device)

                    embs = []
                    for (b,d,h,w) in all_coords:
                        f_patch = decoder_feats[b, :, d-10:d+10, h-10:h+10, w-10:w+10]  # [16,20,20,20]
                        l_patch = pred_logits[b, :, d-10:d+10, h-10:h+10, w-10:w+10]      # [1,20,20,20]
                        v_feat = f_patch.view(16, -1).mean(dim=1)
                        l_feat = l_patch.view(1, -1).mean(dim=1)
                        embs.append(torch.cat([v_feat, l_feat], dim=0))
                    E = torch.stack(embs, dim=0)  # [2P,17]
                
                    # project & normalize
                    Z = F.normalize(self.proj_head(E), dim=1)  # define proj_head in __init__
                
                    # contrastive loss
                    loss_con = self.contrastive_loss(Z, labels)
                else:
                    loss_con = torch.tensor(0.).cuda()

                con_loss_test += loss_con.item()

                total_loss += dice_loss.item() + seg_ce_loss.item() + loss_con.item() # + ce_loss.item()
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
        avg_con_loss = con_loss_test/len(self.test_dataloader)
        #avg_ce_loss = ce_loss_test/len(self.test_dataloader)

        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest Loss: {average_loss}')
        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest Pixel-based BCE Loss: {avg_segce_loss}')
        #print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest BCE Loss: {avg_ce_loss}')
        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest Dice Loss: {avg_dice_loss}')
        print(f'Iteration:{cur_iter+1}/{self.args.total_iters}\tTest Contrastive Loss: {avg_con_loss}')
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

