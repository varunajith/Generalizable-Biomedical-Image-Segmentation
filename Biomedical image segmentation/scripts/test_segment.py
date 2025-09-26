# Import necessary libraries and packages
import os
import torch.utils.data as data
import transforms as transforms
import numpy as np
import argparse
import random
from loss import CombinedLoss
from data import BasicDataset,get_image_mask_pairs  # Import the Dataset handling module
import torch
from evaluate2 import evaluate_segmentation  # Import the evaluation function
from unet import DCUNet #HalfUNet#, UNetPlusPlus

import sys

# Simulate command-line arguments

sys.argv = ['test_segment.py', '--seg_model','UNet','--test_set_dir', 'data/S1Real140/test',
            '--seg_ckpt_dir', 'DCUNet_S1Real140_Diceloss_lr0.0001/Best_Model_Checkpoint.pth',
            '--output_dir', 'DCUNet_S1Real140_Diceloss_lr0.0001/Best_Model_Checkpoint', '--scale', '1',
            '--classes', '2']
              

# Set up seeds for reproducible results
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# Define the testing function
def test(args, image_size=[512, 768], image_means=[0.5], image_stds=[0.5], batch_size=1):
    # Determine if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the transformation to be applied on the images
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_means, std=image_stds)
    ])

    # Read samples
    sample_pairs = get_image_mask_pairs(args.test_set_dir)
    assert len(sample_pairs)>0, f'No samples found in {args.test_set_dir}'

    # Load the test dataset and apply the transformations
    test_data = BasicDataset(sample_pairs,transforms=test_transforms)

    # Create a dataloader for the test dataset
    test_iterator = data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print(test_iterator)

    # Create an instance of the Segmentation model and load the trained model
    if args.seg_model == 'UNet':
        #Seg = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
        #Seg = UNetPlusPlus(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
        #Seg = HalfUNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
        Seg = DCUNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    else:
        # If none of the above models are matched, raise an error
        raise ValueError(f"Model '{args.seg_model}' not found.")
    
    # Move the model to the GPU
    Seg = Seg.to(device)
    Seg.load_state_dict(torch.load(args.seg_ckpt_dir))
    

    # Define the loss functions
    Seg_criterion = CombinedLoss()

    # Evaluate the model and calculate the dice score and average precision
    scores = evaluate_segmentation(Seg, test_iterator, device,Seg_criterion, len(test_data),
                                                           is_avg_prec=True, prec_thresholds=[0.5,0.6,0.7,0.8,0.9],
                                                           output_dir=args.output_dir)

    # Save metrics
    with open(os.path.join(args.output_dir, 'Seg_metrics.txt'), 'w') as f:
        f.write(f"""Average Dice score: {scores['dice_score']}
Average loss: {scores['avg_val_loss']}
Average precision at ordered thresholds: {scores['avg_precision']}
Average recall at ordered thresholds: {scores['avg_recall']}
Average fscore at ordered thresholds: {scores['avg_fscore']}""")
    return scores

# Define the main function
if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument("--test_set_dir", required=True, type=str, help="path for the test dataset")
    parser.add_argument("--seg_ckpt_dir", required=True, type=str, help="path for the checkpoint of segmentation model to test")
    parser.add_argument("--output_dir", required=True, type=str, help="path for saving the test outputs")
    parser.add_argument("--seg_model", required=True, type=str, help="segmentation model type (DeepSea or CellPose or UNET)")
    parser.add_argument('--viz', '-v', action='store_true',
                       help='Visualize the images as they are processed')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=True, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Check if the test set directory exists
    assert os.path.isdir(args.test_set_dir), 'No such file or directory: ' + args.test_set_dir

    # Run the test function
    scores=test(args)

    # Print scores
    print('Average Dice score:', scores['dice_score'])
    print('Average loss:', scores['avg_val_loss'])
    print('Average precision at ordered thresholds:', scores['avg_precision'])
    print('Average recall at ordered thresholds:', scores['avg_recall'])
    print('Average fscore at ordered thresholds:', scores['avg_fscore'])
    