import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pdb
import torch.nn.functional as F
import nibabel as nib
import pandas as pd

def paired_loader(args):
      train_dataset = MRIDataset(args.data_path, mode="train")
      val_dataset = MRIDataset(args.data_path, mode="val")
      test_dataset = MRIDataset(args.data_path, mode="test")

      print("===> Total size of paired train set " + str(len(train_dataset)))
      print("===> Total size of paired test set " + str(len(test_dataset)))

      train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, drop_last=True)
      val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, drop_last=False)

      test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, drop_last=False)

      return train_data_loader, val_data_loader, test_data_loader

def paired_loader_patch(args):
      train_dataset = MRIDatasetSub(args.data_path, mode="train")
      val_dataset = MRIDatasetSub(args.data_path, mode="val")
      test_dataset = MRIDatasetSub(args.data_path, mode="test")

      print("===> Total size of paired train set " + str(len(train_dataset)))
      print("===> Total size of paired test set " + str(len(test_dataset)))

      train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, drop_last=True)
      val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, drop_last=False)

      test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, drop_last=False)

      return train_data_loader, val_data_loader, test_data_loader


def combine_subvolumes(tokens, original_shape, subvolume_size):
     """
     Combine sub-volumes back into the original spatial shape.
     
     Args:
         tokens: Tensor of shape (num_subvolumes, C, d, h, w)
         original_shape: Tuple (B, C, D, H, W) of the original input shape
         subvolume_size: Tuple (d, h, w) of the sub-volume size
 
     Returns:
         Tensor of shape (B, C, D, H, W)
     """
     if len(original_shape) == 5:
         B, C, D, H, W = original_shape
         d, h, w = subvolume_size
 
         num_d = D // d
         num_h = H // h
         num_w = W // w

         # Reshape tokens back into sub-volume grid
         tokens = tokens.view(B, num_d, num_h, num_w, C, d, h, w)
         tokens = tokens.permute(0, 4, 1, 5, 2, 6, 3, 7)  # Rearrange axes
         return tokens.contiguous().view(B, C, D, H, W)
     elif len(original_shape) == 4:
         # Handle 4D input (B, D, H, W)
         B, D, H, W = original_shape
         d, h, w = subvolume_size
 
         num_d = D // d
         num_h = H // h
         num_w = W // w
 
         # Reshape tokens back into sub-volume grid
         tokens = tokens.view(B, num_d, num_h, num_w, d, h, w)
         tokens = tokens.permute(0, 1, 4, 2, 5, 3, 6)  # Rearrange axes
         return tokens.contiguous().view(B, D, H, W)
     else:
     	raise ValueError(f"Unsupported original shape: {original_shape}. Expected 4D or 5D tensor.")


def divide_into_subvolumes(x, subvolume_size):
    """
    Divide a 3D volume into sub-volumes.
 
     Args:
         x: Input tensor of shape (B, C, D, H, W)
         subvolume_size: Tuple (d, h, w) specifying the size of each sub-volume.
 
     Returns:
        Tensor of shape (B * num_subvolumes, C, d, h, w)
    """
    if len(x.shape) == 5:
        B, C, D, H, W = x.shape
        d, h, w = subvolume_size

        # Reshape the input into sub-volumes
        x = x.unfold(2, d, d).unfold(3, h, h).unfold(4, w, w)
        x = x.contiguous().view(-1, C, d, h, w)  # Flatten the sub-volume dimension into batch
        return x
    elif len(x.shape) == 4:
        # Case where input is of shape (B, D, H, W)
        B, D, H, W = x.shape
        d, h, w = subvolume_size

        # Reshape the input into sub-volumes
        x = x.unfold(1, d, d).unfold(2, h, h).unfold(3, w, w)
        x = x.contiguous().view(-1, d, h, w)  # Flatten the sub-volume dimension into batch
        return x
    else:
        raise ValueError(f"Unsupported input shape: {x.shape}. Expected 4D or 5D tensor.")


class MRIDataset(Dataset):
    def __init__(self, data_path, mode="train", transform=None):
        self.data_path = data_path
        self.mode = mode
        self.transform = transform

        if mode == "train":
            df_data = pd.read_csv(os.path.join(self.data_path, "train.csv"))
        elif mode == "test":
            df_data = pd.read_csv(os.path.join(self.data_path, "test.csv"))
        
        self.pair_list = df_data.to_dict("records")

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, index):
        cur_item = self.pair_list[index]

    	# load the data
        img_path = cur_item["MRI_file_path"]
        mask_path = cur_item["GT_mask_path"]
        #hdog_path = cur_item["HDoG_file_path"]
        #cmb_label_vol = cur_item["CMB_label"]
        
        image = nib.load(os.path.join(self.data_path, img_path)).get_fdata()
        img_max, img_min = image.max(), image.min()
        # normaliza image to 0-1 range
        image = (image - img_min)/(img_max - img_min)
        mask = nib.load(os.path.join(self.data_path, mask_path)).get_fdata()

        if self.transform:
            image, mask = self.transform(image), self.transform(mask)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        cmb_label_vol = (mask.sum() > 0).float()

        return img_path, image, mask, cmb_label_vol


class MRIDatasetSub(Dataset):
    def __init__(self, data_path, mode="train", sub_size = (64, 64, 48), transform=None):
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.sub_size = sub_size

        if mode == "train":
            df_data = pd.read_csv(os.path.join(self.data_path, "train.csv"))
        elif mode == "val":
            df_data = pd.read_csv(os.path.join(self.data_path, "val.csv"))
        elif mode == "test":
            df_data = pd.read_csv(os.path.join(self.data_path, "test.csv"))
        
        self.pair_list = df_data.to_dict("records")

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, index):
        cur_item = self.pair_list[index]

    	# load the data
        img_path = cur_item["MRI_file_path"]
        mask_path = cur_item["GT_mask_path"]
        #hdog_path = cur_item["HDoG_file_path"]
        #cmb_label_vol = cur_item["CMB_label"]
        image = nib.load(os.path.join(self.data_path,img_path)).get_fdata()
        #print("Shape of the raw data:"+str(image.shape))
        mask = nib.load(os.path.join(self.data_path,mask_path)).get_fdata()
        #hdog = nib.load(os.path.join(self.data_path,hdog_path)).get_fdata()

        # normalize image data to 0-1
        image = (image - image.min())/(image.max() - image.min())


        if self.transform:
            image, mask = self.transform(image), self.transform(mask)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        image_subV = divide_into_subvolumes(image, self.sub_size)
        #hdog = torch.tensor(hdog, dtype=torch.float32).unsqueeze(0)
        #hdog_subV = divide_into_subvolumes(hdog, self.sub_size)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        mask_subV = divide_into_subvolumes(mask, self.sub_size)

        # Compute patch-level ground truth labels:
        # Label = 1 if any voxel in the mask patch is positive, else 0.
        patch_labels = (mask_subV.view(mask_subV.size(0), -1).sum(dim=1) > 0).float()

        return img_path, image_subV, mask_subV, patch_labels
