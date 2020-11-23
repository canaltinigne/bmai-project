"""
Training Script for BmAI Project
@author: Can Altinigne

This script is the main program that trains BmAI models.
Please carefully check the commented lines. I might have
commented some losses as I did not use them in the training
in some experiments. In order to run the training with other 
losses, please consider uncommenting those lines.

    - Parameters:
        - e: Number of epochs
        - l: Learning rate
        - ls: Loss Function for Height and Weight Estimation, Please 
              check the if section on args.loss to see supported loss
              functions.
        - u: Consider only upper body parts for Lying dataset.
        - m: Number of neurons in the first convolution layer of U-Net.   
        - bs: Batch size.
        - pr: Initialize the model with pretrained weights from our paper.
        - d: Dataset name, ['lying', 'standing', 'shallow'].

"""

import numpy as np 
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader

from loss import dice_coef, dice_loss
from dataset import HMP_Dataset, HMP_Lying_Dataset, ShallowDataset
from network import UNet, ShallowNet
from datetime import datetime
from glob import glob
from torch.nn.parameter import Parameter

"""
Example Run:
CUDA_VISIBLE_DEVICES=1 python train.py -e 50 -l 1e-3 -ls mse -d [DATASET NAME] -pr 0
"""
    
if __name__ == "__main__":
    
    # PARSER SETTINGS
    parser = argparse.ArgumentParser(description='U-Net PyTorch Model for Height and Weight Prediction in IMDB Dataset')

    parser.add_argument('-e', '--epoch', type=int, required=True, help='Number of Epochs')
    parser.add_argument('-l', '--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('-ls', '--loss', type=str, required=True, help='Height/Weight loss type')
    parser.add_argument('-u', '--upper_body', type=int, required=True, help='Consider upper body only for lying kids')
    
    parser.add_argument('-m', '--min_neuron', type=int, default=128, help='Minimum neuron number for the first layer')
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('-pr', '--pretrained', type=int, required=True, help='Load pretrained weights')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Dataset name')

    args = parser.parse_args()

    # INITIALIZATIONS
    n_epochs = args.epoch
    
    if args.loss == 'mse':
        height_loss = nn.MSELoss()
        weight_loss = nn.MSELoss()
    elif args.loss == 'mae':
        height_loss = nn.L1Loss()
        weight_loss = nn.L1Loss()
    elif args.loss == 'huber':
        height_loss = nn.SmoothL1Loss()
        weight_loss = nn.SmoothL1Loss()
        
    
    if args.dataset == 'lying':
        train = DataLoader(HMP_Lying_Dataset('train', upper=args.upper_body), 
                           batch_size=args.batch_size, num_workers=8, shuffle=True)
        
        valid = DataLoader(HMP_Lying_Dataset('val', upper=args.upper_body), 
                           batch_size=1, num_workers=8, shuffle=False)
    
    elif args.dataset == 'standing':
        train = DataLoader(HMP_Dataset('train'), batch_size=args.batch_size, num_workers=8, shuffle=True)
        valid = DataLoader(HMP_Dataset('val'), batch_size=1, num_workers=8, shuffle=False)
    
    elif args.dataset == 'shallow':
        train = DataLoader(ShallowDataset('train'), batch_size=args.batch_size, num_workers=8, shuffle=True)
        valid = DataLoader(ShallowDataset('val'), batch_size=1, num_workers=8, shuffle=False)

        
    print("Training on " + str(len(train)*args.batch_size) + " images.")
    print("Validating on " + str(len(valid)) + " images.")

    #net = UNet(args.min_neuron)
    net = ShallowNet()
    
    start_epoch = 0
    
    if args.pretrained:
        load_model = torch.load('/data/HMP_Project/models/MODEL_16072020_172757_pretrained1/model_ep_51.pth.tar')["state_dict"]
        net.load_state_dict(load_model, strict=False)
                        
    SAVE_DIR = 'SHALLOW_MODEL_' + datetime.now().strftime("%d%m%Y_%H%M%S") + '_pretrained{}/'.format(args.pretrained)
    
    MODEL_SETTINGS = {
        'epoch': n_epochs,
        'learning_rate': args.learning_rate,
        'height_loss': args.loss,
        'batch_size': args.batch_size,
        'min_neuron': args.min_neuron,
        'dataset': args.dataset,
        'is_upper': args.upper_body
    }
    
    LOG_DIR = 'logs/' + SAVE_DIR
    
    try:
        os.makedirs(LOG_DIR)
        np.save(LOG_DIR + 'model_settings.npy', MODEL_SETTINGS)
    except:
        print("Error ! Model exists.")
    
    # Print Number of Parameters
    n_params = 0

    for param in net.parameters():
        n_params += param.numel()
        
    print('Total params:', n_params)
    print('Trainable params:', sum(p.numel() for p in net.parameters() if p.requires_grad))
    print('Non-trainable params:',n_params-sum(p.numel() for p in net.parameters() if p.requires_grad))
    
    # Use GPU
    cuda = torch.cuda.is_available()
    if cuda:
        net = net.cuda()
    
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
 
    best_val = np.inf
    best_ep = -1
    
    v_l = []
    v_lh = []
    v_lm = []
    v_lj = []
    v_lw = []
    v_lt = []

    t_l = []
    t_lh = []
    t_lm = []
    t_lj = []
    t_lw = []
    t_lt = []
    
    for ep in range(start_epoch, start_epoch+n_epochs):
        
        net.train()

        with tqdm(total=len(train), dynamic_ncols=True) as progress:
            
            loss_ = 0.
            tm_ = 0.
            tj_ = 0.
            th_ = 0.
            tw_ = 0.
            tt_ = 0.
            
            progress.set_description('Epoch: %s' % str(ep+1))

            for idx, batch_data in enumerate(train):
                #X, y_mask, y_joint, y_height, y_weight, y_tennis = batch_data['img'].cuda(), batch_data['mask'].cuda(), batch_data['joints'].cuda(), batch_data['height'].cuda(), batch_data['weight'].cuda(), batch_data['tennis_ball'].cuda()
                
                X, y_height, y_weight = batch_data['img'].cuda(), batch_data['height'].cuda(), batch_data['weight'].cuda()
                optimizer.zero_grad()
                
                #mask_o, joint_o, height_o, weight_o, tennis_o = net(X)
                height_o, weight_o = net(X)
                                
                #loss_m = (dice_loss(mask_o, y_mask, 0) + dice_loss(mask_o, y_mask, 1))/2
                #loss_j = nn.CrossEntropyLoss()(joint_o, y_joint)  
                loss_h = height_loss(height_o, y_height)
                loss_w = weight_loss(weight_o, y_weight)
                #loss_t = (dice_loss(tennis_o, y_tennis, 0) + dice_loss(tennis_o, y_tennis, 1))/2

                #loss = loss_h + loss_m + loss_j + loss_w + loss_t  
                loss = loss_h + loss_w
                                
                loss.backward()
                optimizer.step()
                
                progress.update(1)
                
                loss_ += loss.item()
                #tm_ += loss_m.item()
                #tj_ += loss_j.item()
                th_ += loss_h.item()
                tw_ += loss_w.item()
                #tt_ += loss_t.item()
                
                progress.set_postfix(loss=loss_/(idx+1), mask=tm_/(idx+1), 
                                     joint=tj_/(idx+1), height=th_/(idx+1), 
                                     weight=tw_/(idx+1), tennis=tt_/(idx+1))

            loss_ /= len(train)
            tm_ /= len(train)
            tj_ /= len(train)
            th_ /= len(train)
            tw_ /= len(train)
            tt_ /= len(train)
            
        progress.write('Validating ...')
        
        net.eval()
        
        with torch.no_grad():
            
            vl_ = 0.
            vm_ = 0.
            vj_ = 0.
            vh_ = 0.
            vw_ = 0.
            vt_ = 0.

            for idx, batch_data in enumerate(valid):
                #X, y_mask, y_joint, y_height, y_weight, y_tennis = batch_data['img'].cuda(), batch_data['mask'].cuda(), batch_data['joints'].cuda(), batch_data['height'].cuda(), batch_data['weight'].cuda(), batch_data['tennis_ball'].cuda()
                X, y_height, y_weight = batch_data['img'].cuda(), batch_data['height'].cuda(), batch_data['weight'].cuda()

                #mask_o, joint_o, height_o, weight_o, tennis_o = net(X)
                height_o, weight_o = net(X)


                #val_loss_m = (dice_loss(mask_o, y_mask, 0) + dice_loss(mask_o, y_mask, 1))/2
                #val_loss_j = nn.CrossEntropyLoss()(joint_o, y_joint)
                val_loss_h = height_loss(height_o, y_height)
                val_loss_w = weight_loss(weight_o, y_weight)
                #val_loss_t = (dice_loss(tennis_o, y_tennis, 0) + dice_loss(tennis_o, y_tennis, 1))/2

                #val_loss = val_loss_h + val_loss_m + val_loss_j + val_loss_w + val_loss_t
                val_loss = val_loss_h + val_loss_w
                
                vl_ += val_loss.item()
                #vm_ += val_loss_m.item()
                #vj_ += val_loss_j.item()
                vh_ += val_loss_h.item()
                vw_ += val_loss_w.item()
                #vt_ += val_loss_t.item()

            vl_ /= len(valid)
            vm_ /= len(valid)
            vj_ /= len(valid)
            vh_ /= len(valid)
            vw_ /= len(valid)
            vt_ /= len(valid)
            
        t_l.append(loss_)
        t_lm.append(tm_)
        t_lj.append(tj_)
        t_lh.append(th_)
        t_lw.append(tw_)
        t_lt.append(tt_)

        v_l.append(vl_)
        v_lm.append(vm_)
        v_lj.append(vj_)
        v_lh.append(vh_)
        v_lw.append(vw_)
        v_lt.append(vt_)

        if vl_ < best_val:

            best_val = vl_

            state = {'epoch': ep + 1, 
                     'state_dict': net.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     't_l': t_l,
                     't_m': t_lm,
                     't_j': t_lj,
                     't_h': t_lh,
                     'v_l': v_l,
                     'v_m': v_lm,
                     'v_j': v_lj,
                     'v_h': v_lh,
                     't_w': t_lw,
                     'v_w': v_lw,
                     't_t': t_lt,
                     'v_t': v_lt
                    }

            if os.path.exists('models/' + SAVE_DIR):
                os.remove('models/' + SAVE_DIR + 'model_ep_{}.pth.tar'.format(best_ep))
            else:
                os.makedirs('models/' + SAVE_DIR)

            torch.save(state, 'models/' + SAVE_DIR + 'model_ep_{}.pth.tar'.format(ep+1))
            best_ep = ep+1
                

        progress.write('T Loss: {:.3f} - T Mask: {:.3f} - T Joint: {:.3f} - T Height: {:.3f} - T Weight: {:.3f} - T Tennis: {:.3f}\nV Loss: {:.3f} - V Mask: {:.3f} - V Joint: {:.3f} - V Height: {:.3f} - V Weight: {:.3f} - V Tennis: {:.3f}'.format(loss_, tm_, tj_, th_, tw_, tt_, vl_, vm_, vj_, vh_, vw_, vt_))
           
   
    state = {'epoch': start_epoch+n_epochs, 
             'state_dict': net.state_dict(),
             'optimizer': optimizer.state_dict(),
             't_l': t_l,
             't_m': t_lm,
             't_j': t_lj,
             't_h': t_lh,
             't_w': t_lw,
             'v_l': v_l,
             'v_m': v_lm,
             'v_j': v_lj,
             'v_h': v_lh,
             'v_w': v_lw,
             't_t': t_lt,
             'v_t': v_lt
    }

    torch.save(state, 'models/' + SAVE_DIR + 'last_model_ep_{}.pth.tar'.format(start_epoch+n_epochs))