import torchvision.utils as vutils
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        targets = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        #inputs = F.softmax(inputs, dim=1)
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        true_pos = (inputs * targets).sum()
        false_neg = (targets * (1 - inputs)).sum()
        false_pos = ((1 - targets) * inputs).sum()
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        return 1 - tversky

def to_img(x, name='img', resize=-1, normalize=True, n_row=None):
    if len(x.shape) < 4:
        x = x.unsqueeze(0)

    if normalize:
        # x = ((x - x.min()) / (x.max() - x.min()))
        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)

    if resize > 0:
        transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(size=resize), transforms.ToTensor()])
        x = [transform(x_) for x_ in x]
        x = torch.stack(x)

    if n_row is None:
        n_row = int(x.shape[0] ** 0.5)

    x = vutils.make_grid(x, normalize=False, nrow=n_row)
    vutils.save_image(x, name + '.png')


def adjust_learning_rate(optimizer, lr, i, total_iters, warmup):
    if i < warmup:
        new_lr = lr * (i + 1) / (warmup + 1)
    elif i >= total_iters // 2:
        half_iters = total_iters // 2
        new_lr = (1 - (i - half_iters) / half_iters) * lr
    else:
        new_lr = lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr

def cyclical_learning_rate(optimizer, batch_step,
                           step_size,
                           base_lr=0.001,
                           max_lr=0.005,
                           mode='triangular2',
                           gamma=0.999995):

    cycle = np.floor(1 + batch_step / (2. * step_size))
    x = np.abs(batch_step / float(step_size) - 2 * cycle + 1)

    lr_delta = (max_lr - base_lr) * np.maximum(0, (1 - x))
    
    if mode == 'triangular':
        pass
    elif mode == 'triangular2':
        lr_delta = lr_delta * 1 / (2. ** (cycle - 1))
    elif mode == 'exp_range':
        lr_delta = lr_delta * (gamma**(batch_step))
    else:
        raise ValueError('mode must be "triangular", "triangular2", or "exp_range"')
        
    lr = base_lr + lr_delta
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def iou(mask1, mask2, num_classes=19, smooth=1e-6):
    avg_iou = 0
    for sem_class in range(num_classes):
        pred_inds = (mask2 == sem_class)
        target_inds = (mask1 == sem_class)
        intersection_now = (pred_inds[target_inds]).long().sum().item()
        union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
        avg_iou += (float(intersection_now + smooth) / float(union_now + smooth))
    return (avg_iou / num_classes)


def dice_coeff(mask1, mask2, smooth=1e-6, num_classes=19):
    dice = 0
    dice_bg = 0
    counter=0
    dice_hash = dict()
    for sem_class in range(num_classes):
        pred_inds = (mask2 == sem_class)
        target_inds = (mask1 == sem_class)
        intersection = (pred_inds[target_inds]).long().sum().item()
        denom = pred_inds.long().sum().item() + target_inds.long().sum().item()
        dice_val = (float(2 * intersection) + smooth) / (float(denom) + smooth)
        dice_bg += dice_val
        
        dice_hash[sem_class] = dice_val
        
        if sem_class > 0:
            dice += dice_val
            counter=counter+1

    return dice_bg/num_classes, dice/counter, dice_hash

def dice_loss(mask1, mask2, smooth=1e-6, num_classes=19):

    loss = 0
    counter = 0
    for sem_class in range(num_classes):   # fc5_0.1_10_10
    #for sem_class in range(1, num_classes):
        pred_inds = (mask2 == sem_class)
        target_inds = (mask1 == sem_class)
        intersection = (pred_inds[target_inds]).long().sum().item()
        denom = pred_inds.long().sum().item() + target_inds.long().sum().item()
        dice = (float(2 * intersection) + smooth) / (float(denom) + smooth)
        loss += dice
        counter=counter+1
    return 1 - loss/counter

def dice_loss2(mask1, mask2, smooth=1e-6, num_classes=19):

    loss = 0
    for sem_class in range(1, num_classes):
        pred_inds = (mask2 == sem_class)
        target_inds = (mask1 == sem_class)
        intersection = (pred_inds[target_inds]).long().sum().item()
        denom = pred_inds.long().sum().item() + target_inds.long().sum().item()
        dice = (float(2 * intersection) + smooth) / (float(denom) + smooth)
        loss -= dice
    return loss

def pixelwise_acc(mask1, mask2):
    equals = (mask1 == mask2).sum().item()
    return equals / (mask1.shape[0] * mask1.shape[1] * mask1.shape[2])


def mean_iou(model, dataloader, args):
    gpu1 = args.gpu
    ious = list()
    for i, (data, target) in enumerate(dataloader):
        data, target = data.float().to(gpu1), target.long().to(gpu1)
        prediction = model(data)
        prediction = F.softmax(prediction, dim=1)
        prediction = torch.argmax(prediction, axis=1).squeeze(1)
        ious.append(iou(target, prediction, num_classes=11))

    return (sum(ious) / len(ious))
