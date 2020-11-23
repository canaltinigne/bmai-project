"""
Loss Functions Script
@author: Can Altinigne

This script includes Dice Loss function for image segmentation task.
            
"""

import torch

def dice_coef(input_, target, ch):
    
    """
    The Sorensen–Dice coefficient (Soft Implementation)
    to evaluate intersection between segmentation mask
    and target mask.

    Args:
        input_: Output mask with size [Batch Size, 2, Height, Width].
        target: Target mask with size [Batch Size, 2, Height, Width].
        ch: Selected channel (actually redundant) to take mean for both channels.
        
    Returns:
        - Pixelwise Mean Dice Coefficient between [0, 1]
    """
    
    smooth = 1e-6
    iflat = input_[:,ch,:,:]
    tflat = target[:,ch,:,:]
    intersection = (iflat * tflat).sum(dim=(2,1))
    return torch.mean((2. * intersection + smooth) / (iflat.sum(dim=(2,1)) + tflat.sum(dim=(2,1)) + smooth))


def dice_loss(input_, target, ch):
    
    """
    Loss function to minimize for Dice coefficient maximization.

    Args:
        input_: Output mask with size [Batch Size, 2, Height, Width].
        target: Target mask with size [Batch Size, 2, Height, Width].
        ch: Selected channel (actually redundant) to take mean for both channels.
        
    Returns:
        - (1 - Dice Coefficient)
    """
    
    return 1-dice_coef(input_, target, ch)