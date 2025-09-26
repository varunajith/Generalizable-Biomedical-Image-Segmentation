import torch
import torch.nn.functional as F
from tqdm import tqdm
#from skimage.metrics import structural_similarity as ssim
import numpy as np
from utils.dice_score import multiclass_dice_coeff, dice_coeff


def multiclass_f_score(pred, target, reduce_batch_first=False, epsilon=1e-8):
    # Ensure inputs are binary
    assert pred.size() == target.size()
    
    if pred.dim() == 2 or reduce_batch_first:
        sum_dim = (-1, -2)
    else:
        sum_dim = (-1, -2, -3)

    # Calculate true positives, false positives, and false negatives
    tp = (pred * target).sum(dim=sum_dim)
    fp = (pred * (1 - target)).sum(dim=sum_dim)
    fn = ((1 - pred) * target).sum(dim=sum_dim)

    # Calculate precision and recall
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    # Average F1 score across classes
    return f1.mean()

def calculate_metrics(pred_mask, true_mask):
    """Calculate F1 score and Dice coefficient"""
    pred_mask = pred_mask.astype(np.uint8)
    true_mask = true_mask.astype(np.uint8)

    intersection = np.sum(pred_mask * true_mask)
    union = np.sum(pred_mask) + np.sum(true_mask)
    dice = (2. * intersection) / (union + 1e-7)

    tp = np.sum((pred_mask == 1) & (true_mask == 1))
    fp = np.sum((pred_mask == 1) & (true_mask == 0))
    fn = np.sum((pred_mask == 0) & (true_mask == 1))

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f_score = 2 * (precision * recall) / (precision + recall + 1e-7)

    return f_score.mean(), dice.mean()

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    f_score = 0  # Add F-score tracking

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # Move data to the correct device
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # Predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                # Binary segmentation
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()

                # Calculate Dice score and F-score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                f_score += multiclass_f_score(mask_pred, mask_true.float(), reduce_batch_first=False)
                #f_score += calculate_metrics(mask_pred, mask_true)


            else:
                # Multi-class segmentation
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()

                # Calculate Dice score and F-score, ignoring background (class 0)
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                f_score += multiclass_f_score(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                #f_score += calculate_metrics(mask_pred[:, 1:], mask_true[:, 1:])

                # Calculate SSIM for each image and class (excluding background)


    # Normalize scores by the number of batches
    dice_score /= max(num_val_batches, 1)
    f_score /= max(num_val_batches, 1)
    

    net.train()
    return dice_score, f_score  # Return all scores