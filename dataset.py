import numpy as np 
import cv2
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from keras.utils import to_categorical
from torch.autograd import Variable
import json
from scipy.ndimage import gaussian_filter


class HMP_Dataset(Dataset):

    def __init__(self, mode, res=128):
        
        if mode == 'train':
            self.df = pd.read_csv('TRAINING_SET.csv')
        elif mode == 'val':
            self.df = pd.read_csv('VAL_SET.csv')
        elif mode == 'test':
            self.df = pd.read_csv('TEST_SET.csv')
            
        self.to_tensor = transforms.ToTensor()
        self.res = res
        self.HEIGHT_MEAN = 91.90
        self.WEIGHT_MEAN = 12.54

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        image_name = self.df['img'].iloc[idx]
        mask_name = self.df['mask'].iloc[idx]
        joint_name = self.df['joints'].iloc[idx]
        tennis_mask_name = mask_name.replace('HMP_FRONT_MASKS', 'HMP_TENNIS_MASKS')
        
        height = torch.from_numpy(np.array([self.df['Height_cm'].iloc[idx]])).type(torch.FloatTensor) 
        weight = torch.from_numpy(np.array([self.df['Weight_kg'].iloc[idx]])).type(torch.FloatTensor) 
                
        # Reading Image 
        X = cv2.cvtColor(cv2.imread('./' + image_name), cv2.COLOR_BGR2RGB)
        y_mask = cv2.imread('./' + mask_name).astype('float32') / 255
        y_tennis_mask = cv2.imread('./' + tennis_mask_name).astype('float32') / 255
        
        scale = self.res / max(X.shape[:2])
        scale_tennis = self.res / max(y_tennis_mask.shape[:2])

        X_scaled = cv2.resize(X, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) 
        y_mask_scaled = cv2.resize(y_mask, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        
        
        y_tennis_mask_scaled = cv2.resize(y_tennis_mask, (0,0), fx=scale_tennis, 
                                          fy=scale_tennis, interpolation=cv2.INTER_NEAREST)

        
        l_t = (self.res - y_tennis_mask_scaled.shape[1])//2
        r_t = self.res - y_tennis_mask_scaled.shape[1] - l_t
        
        y_tennis_mask = np.pad(y_tennis_mask_scaled, [(0, 0), (l_t, r_t), (0,0)], mode='constant')
        
        if X_scaled.shape[1] > X_scaled.shape[0]:
            p_a = (self.res - X_scaled.shape[0])//2
            p_b = (self.res - X_scaled.shape[0])-p_a
            X = np.pad(X_scaled, [(p_a, p_b), (0, 0), (0,0)], mode='constant')
            y_mask = np.pad(y_mask_scaled, [(p_a, p_b), (0, 0), (0,0)], mode='constant')
               
        elif X_scaled.shape[1] <= X_scaled.shape[0]:
            p_a = (self.res - X_scaled.shape[1])//2
            p_b = (self.res - X_scaled.shape[1])-p_a
            X = np.pad(X_scaled, [(0, 0), (p_a, p_b), (0,0)], mode='constant') 
            y_mask = np.pad(y_mask_scaled, [(0, 0), (p_a, p_b), (0,0)], mode='constant') 
            
   
        y_mask = y_mask[:,:,0]
        y_tennis_mask = y_tennis_mask[:,:,0]
        
        X = self.to_tensor(X)
        
        # Reading Mask 
        y_mask = to_categorical(y_mask, 2)
        y_mask = self.to_tensor(y_mask)
        
        # Reading Tennis Mask 
        y_tennis_mask = to_categorical(y_tennis_mask, 2)
        y_tennis_mask = self.to_tensor(y_tennis_mask)
        
        # Reading Joint 
        y_heatmap = np.load('./' + joint_name).astype('int64')  # For Heatmaps
        y_heatmap = torch.from_numpy(y_heatmap)

        # Reading Height
        
        return {
            'img': X, 
            'mask': y_mask,
            'joints': y_heatmap,
            'height': height-self.HEIGHT_MEAN, 
            'weight': weight-self.WEIGHT_MEAN,
            'tennis_ball': y_tennis_mask,
            'name': image_name
        }
    

class HMP_Lying_Dataset(Dataset):

    def __init__(self, mode, res=128, rotate=True, upper=False):
        
        if mode == 'train':
            self.df = pd.read_csv('LYING_TRAIN_SET.csv', converters={'ChildID': lambda x: str(x)})
        elif mode == 'val':
            self.df = pd.read_csv('LYING_VAL_SET.csv', converters={'ChildID': lambda x: str(x)})
        elif mode == 'test':
            self.df = pd.read_csv('LYING_TEST_SET.csv', converters={'ChildID': lambda x: str(x)})
            
        self.to_tensor = transforms.ToTensor()
        self.res = res
        self.HEIGHT_MEAN = 71.49
        self.WEIGHT_MEAN = 8.11
        self.rotate = rotate
        self.upper = upper
        
        with open("data/orientation.json", "r") as f:
            self.orientation = json.load(f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        image_name = 'data/LYING_IMAGES/' + self.df['ChildID'].iloc[idx] + '.png'
        
        if self.upper:
            mask_name = image_name.replace('LYING_IMAGES', 'UPPER_BODY_MASKS')
            joint_name = image_name.replace('LYING_IMAGES', 'UPPER_BODY_JOINTS').replace('.png', '.npy')
        else:
            mask_name = image_name.replace('IMAGES', 'MASKS')
            joint_name = image_name.replace('IMAGES', 'JOINTS').replace('.png', '.joint.npy')
             
        tennis_mask_name = image_name.replace('IMAGES', 'TENNIS')
        
        height = torch.from_numpy(np.array([self.df['Height_cm'].iloc[idx]])).type(torch.FloatTensor) 
        weight = torch.from_numpy(np.array([self.df['Weight_kg'].iloc[idx]])).type(torch.FloatTensor)
        
        rot = 0
        
        if self.rotate:
            
            n = self.df['ChildID'].iloc[idx]
            k = self.orientation[n + '_F'] if n + '_F' in self.orientation else self.orientation[n + '-F']
            
            if k == 'L':
                rot = -1
            elif k == 'D':
                rot = 2
            elif k == 'R':
                rot = 1
                
        # Reading Image 
        X = np.rot90(cv2.cvtColor(cv2.imread('./' + image_name), cv2.COLOR_BGR2RGB), k=rot)
        y_mask = np.rot90(cv2.imread('./' + mask_name).astype('float32') / 255, k=rot)
        y_tennis_mask = np.rot90(cv2.imread('./' + tennis_mask_name).astype('float32') / 255, k=rot)
                
        scale = self.res / max(X.shape[:2])
        scale_tennis = self.res / max(y_tennis_mask.shape[:2])

        X_scaled = cv2.resize(X, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) 
        y_mask_scaled = cv2.resize(y_mask, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        
        
        y_tennis_mask_scaled = cv2.resize(y_tennis_mask, (0,0), fx=scale_tennis, 
                                          fy=scale_tennis, interpolation=cv2.INTER_NEAREST)
        
        
        if y_tennis_mask_scaled.shape[1] > y_tennis_mask_scaled.shape[0]:
            l_t = (self.res - y_tennis_mask_scaled.shape[0])//2
            r_t = self.res - y_tennis_mask_scaled.shape[0] - l_t
            y_tennis_mask = np.pad(y_tennis_mask_scaled, [(l_t, r_t), (0,0), (0,0)], mode='constant')

        elif y_tennis_mask_scaled.shape[1] <= y_tennis_mask_scaled.shape[0]:
            l_t = (self.res - y_tennis_mask_scaled.shape[1])//2
            r_t = self.res - y_tennis_mask_scaled.shape[1] - l_t
            y_tennis_mask = np.pad(y_tennis_mask_scaled, [(0, 0), (l_t, r_t), (0,0)], mode='constant')

            
        if X_scaled.shape[1] > X_scaled.shape[0]:
            p_a = (self.res - X_scaled.shape[0])//2
            p_b = (self.res - X_scaled.shape[0])-p_a
            X = np.pad(X_scaled, [(p_a, p_b), (0, 0), (0,0)], mode='constant')
            
            if self.upper == 0:
                y_mask = np.pad(y_mask_scaled, [(p_a, p_b), (0, 0), (0,0)], mode='constant')
               
        elif X_scaled.shape[1] <= X_scaled.shape[0]:
            p_a = (self.res - X_scaled.shape[1])//2
            p_b = (self.res - X_scaled.shape[1])-p_a
            X = np.pad(X_scaled, [(0, 0), (p_a, p_b), (0,0)], mode='constant') 
            
            if self.upper == 0:
                y_mask = np.pad(y_mask_scaled, [(0, 0), (p_a, p_b), (0,0)], mode='constant') 
            
   
        y_mask = y_mask[:,:,0]
        y_tennis_mask = y_tennis_mask[:,:,0]
        
        # Reading Joint 
        y_heatmap = np.rot90(np.load(joint_name).astype('int64'), k=rot)  # For Heatmaps
        y_heatmap = torch.from_numpy(y_heatmap.copy())
 
        X = self.to_tensor(X)
        
        # Reading Mask 
        y_mask = to_categorical(y_mask, 2)
        y_mask = self.to_tensor(y_mask)
        
        # Reading Tennis Mask 
        y_tennis_mask = to_categorical(y_tennis_mask, 2)
        y_tennis_mask = self.to_tensor(y_tennis_mask)

        return {
            'img': X, 
            'mask': y_mask,
            'joints': y_heatmap,
            'height': height-self.HEIGHT_MEAN, 
            'weight': weight-self.WEIGHT_MEAN,
            'tennis_ball': y_tennis_mask,
            'name':  self.df['ChildID'].iloc[idx]
        }
    

class ShallowDataset(Dataset):
    
    def __init__(self, mode):
        
        if mode == 'train':
            self.df = pd.read_csv('TRAINING_SET.csv')
        elif mode == 'val':
            self.df = pd.read_csv('VAL_SET.csv')
        elif mode == 'test':
            self.df = pd.read_csv('TEST_SET.csv')
            
        self.to_tensor = transforms.ToTensor()
        self.HEIGHT_MEAN = 91.90
        self.WEIGHT_MEAN = 12.54
        
        self.part_to_idx = dict(zip([
            'nose',
            'leftEye',
            'rightEye',
            'leftEar',
            'rightEar',
            'leftShoulder',
            'rightShoulder',
            'leftElbow',
            'rightElbow',
            'leftWrist',
            'rightWrist',
            'leftHip',
            'rightHip',
            'leftKnee',
            'rightKnee',
            'leftAnkle',
            'rightAnkle',
        ], range(17)))
    
    
    def __len__(self):
        return len(self.df)
    
    
    def norm(self, x):
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-7)

    
    def __getitem__(self, idx):
        
        image_name = self.df['img'].iloc[idx]
        
        joint_name = 'JOINTS/' + image_name.split('/')[-1].split('.')[0] + '.png.txt'
        body_part_name = image_name.replace('HMP_FRONT_IMAGES', 'BODYPIX_LABELS') + '.txt'
        
        ig = cv2.imread(image_name)
        
        if max(ig.shape) > 640:
            img_size = cv2.resize(ig, None, fx=0.15, fy=0.15).shape
        else:
            img_size = ig.shape

        arr = np.zeros(img_size[0]*img_size[1]).flatten()

        with open(body_part_name, 'r') as f:
            dct = json.load(f)

        for i in dct:
            if dct[i] != -1:
                arr[int(i)] = dct[i]+1

        arr = arr.reshape(img_size[:2])

        scale = 128 / max(arr.shape)
        X_scaled = cv2.resize(arr, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) 

        if X_scaled.shape[1] > X_scaled.shape[0]:
            p_a = (128 - X_scaled.shape[0])//2
            p_b = (128 - X_scaled.shape[0])-p_a
            arr = np.pad(X_scaled, [(p_a, p_b), (0, 0)], mode='constant')

        elif X_scaled.shape[1] <= X_scaled.shape[0]:
            p_a = (128 - X_scaled.shape[1])//2
            p_b = (128 - X_scaled.shape[1])-p_a
            arr = np.pad(X_scaled, [(0, 0), (p_a, p_b)], mode='constant') 
            
            
        body_parts = np.zeros((128,128,24))
        
        for i in np.unique(arr):
            if i != 0:
                body_parts[arr == i, int(i-1)] = 1
            
        with open(joint_name, 'r') as f:
            dct = json.load(f)[0]['keypoints']
            
        keypoints = np.zeros((128,128,17))
        
        for p in dct:
            k_p = np.zeros((128,128))
            k_p[int(p['position']['y'])//4][int(p['position']['x'])//4] = 255
            blurred = self.norm(gaussian_filter(k_p, sigma=2))
            keypoints[...,self.part_to_idx[p['part']]] = blurred
        
        height = torch.from_numpy(np.array([self.df['Height_cm'].iloc[idx]])).float() 
        weight = torch.from_numpy(np.array([self.df['Weight_kg'].iloc[idx]])).float()
        
        return {
            'img': self.to_tensor(np.concatenate([body_parts, keypoints], axis=-1)).float(),
            'height': height-self.HEIGHT_MEAN, 
            'weight': weight-self.WEIGHT_MEAN
        }

    