from torch import Tensor
import torch
import torch.nn.functional as F
from torch.nn import Module
import torch.nn as nn
from torchvision import models
from scipy.ndimage import distance_transform_edt
import numpy as np

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # If input image has 1 channel (grayscale), duplicate it to have 3 channels
        if X.shape[1] == 1:
            X = X.repeat(1, 3, 1, 1)
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class BoundaryDiceLoss(nn.Module):
    def __init__(self, margin=3):
        super().__init__()
        self.margin = margin
        #self.kernel = None
        self.register_buffer('kernel', None)  # Proper buffer registration
        
    def forward(self, inputs, targets,smooth=1e-6):
        # Ensure inputs are 4D and single-channel
        inputs = self._prepare_input(inputs)
        targets = self._prepare_input(targets)
        
        # Initialize kernel if needed
        if self.kernel is None or self.kernel.device != inputs.device:
            self._init_kernel(inputs.device)
            
        # Calculate boundaries
        target_bound = self._get_boundary(targets)
        pred_bound = self._get_boundary(torch.sigmoid(inputs))
        
        # Compute boundary Dice
        intersection = (pred_bound * target_bound).sum()
        denominator = pred_bound.sum() + target_bound.sum()
        return 1 - (2. * intersection + smooth) / (denominator + smooth)
    
    def _prepare_input(self, x):
        """Ensure input is [B,1,H,W]"""
        if x.ndim == 3:  # Handle [B,H,W] case
            return x.unsqueeze(1)
        elif x.shape[1] > 1:  # Multi-channel case
            return x.mean(dim=1, keepdim=True)  # Average across channels
        return x
    
    def _init_kernel(self, device):
        kernel_size = self.margin * 2 + 1
        self.kernel = torch.ones(1, 1, kernel_size, kernel_size,
                               device=device,
                               requires_grad=False) #False
    
    def _get_boundary(self, x):
        """Detect boundary pixels"""
        conv = F.conv2d(x, self.kernel, padding=self.margin)
        
        return (conv > 0) & (conv < self.kernel.sum())




class CombinedLoss(Module):
    def __init__(self,alpha=0.25,gamma=2.0,num_classes=2,dice_weight=1,focal_weight=0,
                 ce_weight=0,t_alpha=0.7,t_beta=0.3,tv_weight=0,delta=1/2,hausdorff_weight=0,
                 sigma=7, W_weight=10, W_focal_weight=0, Bn_dice_weight=0):
        
        ###super().__init__()
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # Weight for the positive class in focal loss
        self.gamma = gamma  # Focusing parameter in focal loss
        self.dice_weight=dice_weight
        self.num_classes = num_classes
        self.focal_weight = focal_weight  # Weight for focal loss term in the combined loss
        self.ce_weight= ce_weight
        self.t_alpha=t_alpha; self.t_beta=t_beta; self.tv_weight=tv_weight;
        self.delta=delta; self.hausdorff_weight=hausdorff_weight
        self.sigma = sigma; self.W_weight=W_weight; self.W_focal_weight=W_focal_weight
        self.Bn_dice_weight=Bn_dice_weight
        
###############################################################################
    def _calculate_boundary_weights(self, mask):
        """
        Calculates the boundary weight map for a given binary mask.

        Args:
            mask (torch.Tensor): Binary mask (1 for object, 0 for background)
                                 of shape (B, H, W).

        Returns:
            torch.Tensor: Boundary weight map of shape (B, 1, H, W).
        """
        batch_size, height, width = mask.shape
        boundary_weights = torch.zeros_like(mask, dtype=torch.float32)

        for b in range(batch_size):
            binary_mask_np = mask[b].cpu().numpy().astype(bool)
            if np.any(binary_mask_np):
                # Compute distance to the nearest boundary
                distance = distance_transform_edt(binary_mask_np)
                boundary = (distance == 0).astype(float)
                dt = distance_transform_edt(1 - boundary)

                # Apply Gaussian weighting
                boundary_weight = self.W_weight * np.exp(-(dt**2) / (2 * (self.sigma**2)))
                boundary_weights[b] = torch.from_numpy(boundary_weight).to(mask.device)

        return boundary_weights.unsqueeze(1)  # Add channel dimension
    
    

    def weighted_cross_entropy_loss(self, inputs, targets):
        """
        Computes the weighted cross-entropy loss without using torch.long.

        Args:
            inputs (torch.Tensor): Model output logits, shape (B, C, H, W).
            targets (torch.Tensor): Ground truth masks, shape (B, H, W).

        Returns:
            torch.Tensor: Weighted cross-entropy loss.
        """
        # Ensure targets are in the range [0, 1] for binary segmentation
        if torch.max(targets) > 1 or torch.min(targets) < 0:
            raise ValueError("Targets must be in the range [0, 1] for binary segmentation.")

        # Calculate class weights (inverse frequency)
        class_counts = torch.sum(targets, dim=(1, 2))  # Count of class 1 (foreground)
        class_weights = 1.0 / (class_counts.float() + 1e-6)  # Add epsilon for stability
        class_weights = class_weights / class_weights.sum()  # Normalize

        # Calculate boundary weights
        boundary_weights = self._calculate_boundary_weights(targets)

        # Combine class weights and boundary weights
        total_weights = class_weights.view(-1, 1, 1) * boundary_weights
        
        # Boundary loss
        pred_bound = torch.sigmoid(10*(inputs - 0.5))  # Sharpened boundaries
        target_bound = torch.sigmoid(10*(targets - 0.5))
        
        # Compute binary cross-entropy loss
        #bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        bce_loss = F.binary_cross_entropy_with_logits(pred_bound, target_bound, reduction='none')
        
        pt = torch.exp(-bce_loss)
        focal_ce_loss = (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()
        
        weighted_loss = (bce_loss * total_weights).mean()
        weighted_focal_ce_loss = (focal_ce_loss*total_weights).mean()

        return weighted_loss, weighted_focal_ce_loss
        

##########################################################################

    def forward(self, inputs, targets, smooth=1e-6):
        
        # Compute Cross-Entropy (CE) loss
        ce_loss = F.cross_entropy(inputs, targets)
        
        # Compute Dice loss
        inputs = F.softmax(inputs, dim=1)[:, 1]
        targets = (targets == 1).float()  # One-hot encoding
                  
        intersection = (inputs * targets).sum()
        dice_loss = 1 - ((2. * intersection + smooth) /
                         (inputs.sum() + targets.sum() + smooth))
        #focal_loss
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Focal-Tversky Loss
        true_positive = (inputs * targets).sum()
        false_positive = ((1 - targets) * inputs).sum()
        false_negative = (targets * (1 - inputs)).sum()
        
        tversky_index = true_positive / (true_positive + self.t_alpha * false_positive + self.t_beta * false_negative)
        tversky_loss = 1 - tversky_index  # Tversky loss is 1 - Tversky index
        Focal_tversky_loss=tversky_loss**(1/self.delta)
        
        # Hausdorff Distance Loss (Differentiable Approximation)
        hausdorff_loss = self.hausdorff_distance_loss(inputs, targets)
        
        # Weighted Cross-Entropy Loss + focal
        weighted_loss,weighted_focal_ce_loss = self.weighted_cross_entropy_loss(inputs, targets)
         
        #Boundary-Aware Dice (BADice)
        boundary_dice = BoundaryDiceLoss(margin=3)
        boundary_dice_loss = boundary_dice(inputs,targets).mean()
        
        # Combine the losses
        combined_loss=(self.dice_weight*dice_loss
                       +self.focal_weight*focal_loss
                       +self.ce_weight*ce_loss
                       +self.tv_weight * Focal_tversky_loss
                       +self.hausdorff_weight*hausdorff_loss
                       +self.dice_weight*dice_loss
                       +self.W_focal_weight*weighted_focal_ce_loss
                       +self.Bn_dice_weight*boundary_dice_loss
                       )
        
        return combined_loss
    
    
    def hausdorff_distance_loss(self, pred, target):
        """
        Differentiable approximation of Hausdorff Distance using distance transforms.
        """
        # Convert tensors to numpy for distance transform (retain gradients)
        device = pred.device
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        batch_size = pred.shape[0]
        total_loss = 0.0

        for i in range(batch_size):
            # Compute distance transform for target boundaries
            target_dt = distance_transform_edt(1 - target_np[i])  # (H, W)
            target_dt = torch.from_numpy(target_dt).float().to(device)

            # Compute distance transform for predicted boundaries
            pred_dt = distance_transform_edt(1 - pred_np[i])
            pred_dt = torch.from_numpy(pred_dt).float().to(device)

            # Calculate bidirectional distances (pred -> target and target -> pred)
            loss_pred_to_target = torch.mean(pred[i] * target_dt)  # Penalize predictions far from target
            loss_target_to_pred = torch.mean((1 - pred[i]) * pred_dt)  # Penalize missed target regions

            # Use sum or max of the two terms as the Hausdorff approximation
            total_loss += (loss_pred_to_target + loss_target_to_pred) / 2.0  # Average bidirectional loss

        return total_loss / batch_size
   


######################################################
class LearningRateScheduler:
    def __init__(self, model, patience=50, factor=0.02, min_lr=1e-6,max_t_alpha=0.98):
        """
        Scheduler to adjust t_alpha and t_beta based on F-score performance.
        
        :param model: The model containing t_alpha and t_beta parameters.
        :param patience: Number of epochs to wait before adjusting t_alpha and t_beta.
        :param factor: Factor by which to reduce the learning rate when plateaus.
        :param min_lr: Minimum learning rate to avoid going to zero.
        """
        self.model = model
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.avg_fscore_best = -float('inf')  # Best F-score seen so far
        self.counter = 0  # Counter for epochs without improvement
        self.max_t_alpha=max_t_alpha

    def step(self, avg_fscore):
        """
        Update the learning rates of t_alpha and t_beta based on the average F-score.
        
        :param avg_fscore: The current average F-score.
        """
        # If the F-score has improved, reset the counter
        if avg_fscore > self.avg_fscore_best:
            self.avg_fscore_best = avg_fscore
            self.counter = 0
        else:
            # If no improvement, increase the counter
            self.counter += 1

            # If the patience has been exceeded, reduce the learning rate for t_alpha and t_beta
            if self.counter >= self.patience:
                # Reduce learning rates by the factor
                if self.t_alpha <= self.max_t_alpha:
                   self.model.t_alpha = self.model.t_alpha + self.factor
                   self.model.t_beta = self.model.t_beta - self.factor
                   
                elif self.t_alpha >= self.max_t_alpha:
                    self.t_alpha=self.max_t_alpha


                print(f"Reducing t_alpha to {self.model.t_alpha}, t_beta to {self.model.t_beta}")

                # Reset the counter after reducing the learning rate
                self.counter = 0

    def get_current_lr(self):
        return self.model.t_alpha, self.model.t_beta


######################################################    
def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]