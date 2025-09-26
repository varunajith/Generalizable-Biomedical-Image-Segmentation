import argparse
import logging
import os
import random
import numpy as np
import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torchvision.transforms as transforms
#import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
#from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
#import logging
#import wandb
#from evaluate import evaluate

from unet import DCUNet #,HalfUNet #UNet, UNetPlusPlus

#from utils.data_loading import CarvanaDataset
#from utils.dice_score import dice_loss
from evaluate2 import evaluate_segmentation
from loss import CombinedLoss
import transforms as transforms
from data import BasicDataset,get_image_mask_pairs
import torch.utils.data as data
import sys

train_set_dir = 'data/S1Real140/train'


dir_checkpoint = Path('./DCUNet_S1Real140_Diceloss_lr0.0001/')

sys.argv = ['traib.py','--train_set_dir',train_set_dir,
            '--epochs','1000','--learning-rate','0.0001','--batch-size','2','--scale','1',
            '--classes','2','--p_vanilla','0.3']

# Set a constant seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Function to reset logging configuration
def reset_logging():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
def train_model(
        model,
        device,
        epochs: int = 3000,
        batch_size: int = 2,
        learning_rate: float = 1e-7,
        val_percent: float = 0.2,
        save_checkpoint: bool = True,
        img_scale: float = 1,
        amp: bool = True,
        weight_decay: float = 1e-6,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        image_size = [512,768],
        image_means = [0.5],image_stds= [0.5]
):
    # 1. Create dataset and augmentation

    train_transforms = transforms.Compose([
                                   transforms.ToPILImage(),
                                   transforms.RandomApply([transforms.RandomOrder([
                                       transforms.RandomApply([transforms.ColorJitter(brightness=0.33, contrast=0.33, saturation=0.33, hue=0)],p=0.5),#0.33,0.33,0.33,0,0.5
                                       transforms.RandomApply([transforms.GaussianBlur((5, 5), sigma=(0.1, 1.0))],p=0.5),#(5,5),(0.1,1.0)
                                       transforms.RandomApply([transforms.RandomHorizontalFlip(0.5)],p=0.5),#(0.5,0.5)
                                       transforms.RandomApply([transforms.RandomVerticalFlip(0.5)],p=0.5),#(0.5,0.5)
                                       transforms.RandomApply([transforms.AddGaussianNoise(0., 0.01)], p=0.5),#(0.5,0.5)
                                       transforms.RandomApply([transforms.CLAHE()], p=0.5),#(0.5)
                                       transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.5),#2,0.5
                                       transforms.RandomApply([transforms.RandomCrop()], p=0.5),#0.5
                                    ])],p=args.p_vanilla),
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean = image_means,std = image_stds)
                               ])

    dev_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_means,std=image_stds)
        ])
        # Read samples

        #dataset = BasicDataset(dir_img, dir_mask, img_scale)
        
   # Read samples
    sample_pairs=get_image_mask_pairs(args.train_set_dir)
    assert len(sample_pairs)>0, f'No samples found in {args.train_set_dir}'
    train_ratio = 1 -  val_percent
    # Split samples
    train_sample_pairs=sample_pairs[:int(train_ratio*len(sample_pairs))]
    valid_sample_pairs=sample_pairs[int(train_ratio * len(sample_pairs)):]

    # Define the datasets for training and validation
    train_data = BasicDataset(train_sample_pairs,transforms=train_transforms,vanilla_aug=args.p_vanilla,gen_nc=1)
    valid_data = BasicDataset(valid_sample_pairs,transforms=dev_transforms,gen_nc=1)

    # Define the dataloaders
    train_loader = data.DataLoader(train_data,shuffle=True,batch_size = args.batch_size,num_workers=8,pin_memory=True)
    val_loader = data.DataLoader(valid_data,shuffle=True,batch_size = 1,num_workers=8 ,pin_memory=True)
   
    # 2. Split into train / validation partitions
    n_val = int(len(sample_pairs) * val_percent)
    n_train = len(sample_pairs) - n_val
      
    # Reset logging configuration
    reset_logging()

    # Set up the logging
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=os.path.join(dir_checkpoint, 'train.log'), filemode='w',
                        format='%(asctime)s - %(message)s', level=logging.INFO)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(model.parameters(), #Deafult optimizer RMSprop
                              lr=learning_rate, weight_decay=weight_decay,betas=(0.9,0.999), foreach=True)
    #optimizer = optim.RMSprop(model.parameters(),
                             #lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=90,factor=0.75)  # goal: maximize Dice score
    grad_scaler = torch.amp.GradScaler('cuda',enabled=amp)
    
    global_step = 0
    # Track the best F1 score
    best_f1_score = 0.0
    
    
    Seg_criterion = CombinedLoss()
    Seg_criterion = Seg_criterion.to(device)
    
    # 5. Begin training
    logging.info('>>>> Start training')
    print('INFO: Start training ...')
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            #for batch in train_loader:
            #for step,batch in enumerate(tqdm(train_loader)):
            for step,batch in enumerate(train_loader):
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                
                
                loss = 0
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss += Seg_criterion(masks_pred, torch.squeeze(true_masks.to(dtype=torch.long), dim=1))
                    else:
                        loss += Seg_criterion(masks_pred, torch.squeeze(true_masks.to(dtype=torch.long), dim=1))
                        
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                
                global_step += 1
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                        
                        
        scores = evaluate_segmentation(model, val_loader, device,Seg_criterion,len(val_loader),is_avg_prec=True,prec_thresholds=[0.5],output_dir='val_itr_eval_seg')
        Val_Dice_Score = scores['dice_score'].item(); Val_f1_Score = scores['avg_fscore'].item()                      
                        
        scheduler.step(best_f1_score)

        logging.info(f"Epoch: {epoch}/{epochs}, Dice score: {Val_Dice_Score:.10f}, f_score: {Val_f1_Score:.10f}")#Training Dice score: {Train_Dice_Score:.10f}, Training f1 score: {Train_f1_Score:.10f},Training SSIM score: {Train_ssim_score:.10f}, Validation SSIM score: {Val_ssim_score:.10f}")


        if Val_f1_Score > best_f1_score:
            best_f1_score = Val_f1_Score
            if save_checkpoint:
                #Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                torch.save(state_dict, str(dir_checkpoint / 'Best_Model_Checkpoint.pth'))
                logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--train_set_dir',required=True,type=str,default=train_set_dir,help="path for the train dataset")
    #parser.add_argument('--dir_checkpoint',required=True,type=str,default=dir_checkpoint,help="path for log_dir and checkpoint")
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=True, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--p_vanilla', '-p', type=float, default=0.2, help='augmentation')
    

    return parser.parse_args()


if __name__ == '__main__':
    torch.cuda.empty_cache()
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    #model = UNetPlusPlus(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    #model = HalfUNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    model = DCUNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
