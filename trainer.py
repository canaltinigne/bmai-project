"""
Experimental Trainer Functions
@author: Can Altinigne

I implemented these functions for more modular codebase.
These functions are still experimental, but you can integrate
train and evaluate function in train.py to get a more modular
training code.

"""

def train(net, train_loader):
    
    """
    Train function.
        
    Args:
        net: Network object.
        train_loader: DataLoader object for the training set.
    
    Returns:
        Not defined yet.
        
    """
    
    net.train()
    
    with tqdm(total=len(train), dynamic_ncols=True) as progress:

        loss_ = 0.
        tm_ = 0.
        tj_ = 0.
        th_ = 0.
        tw_ = 0.

        progress.set_description('Epoch: %s' % str(ep+1))

        for idx, batch_data in enumerate(train):
            X, y_mask, y_joint, y_height, y_weight = batch_data['img'].cuda(), batch_data['mask'].cuda(), batch_data['joints'].cuda(), batch_data['height'].cuda(), batch_data['weight'].cuda()

            optimizer.zero_grad()

            mask_o, joint_o, height_o, weight_o = net(X)

            loss_m = (dice_loss(mask_o, y_mask, 0) + dice_loss(mask_o, y_mask, 1))/2
            loss_j = nn.CrossEntropyLoss()(joint_o, y_joint)  
            loss_h = height_loss(height_o, y_height)
            loss_w = weight_loss(weight_o, y_weight)

            loss = loss_h + loss_m + loss_j + loss_w  

            loss.backward()
            optimizer.step()

            progress.update(1)

            loss_ += loss.item()
            tm_ += loss_m.item()
            tj_ += loss_j.item()
            th_ += loss_h.item()
            tw_ += loss_w.item()

            progress.set_postfix(loss=loss_/(idx+1), mask=tm_/(idx+1), 
                                 joint=tj_/(idx+1), height=th_/(idx+1), 
                                 weight=tw_/(idx+1))

        loss_ /= len(train)
        tm_ /= len(train)
        tj_ /= len(train)
        th_ /= len(train)
        tw_ /= len(train)


def validate(net, valid_loader):
    
    """
    Validation function.
        
    Args:
        net: Network object.
        train_loader: DataLoader object for the validation set.
    
    Returns:
        Not defined yet.
        
    """
    
    net.eval()
        
    with torch.no_grad():

        vl_ = 0.
        vm_ = 0.
        vj_ = 0.
        vh_ = 0.
        vw_ = 0.

        for idx, batch_data in enumerate(valid):
            X, y_mask, y_joint, y_height, y_weight = batch_data['img'].cuda(), batch_data['mask'].cuda(), batch_data['joints'].cuda(), batch_data['height'].cuda(), batch_data['weight'].cuda()

            mask_o, joint_o, height_o, weight_o = net(X)

            val_loss_m = (dice_loss(mask_o, y_mask, 0) + dice_loss(mask_o, y_mask, 1))/2
            val_loss_j = nn.CrossEntropyLoss()(joint_o, y_joint)
            val_loss_h = height_loss(height_o, y_height)
            val_loss_w = weight_loss(weight_o, y_weight)

            val_loss = val_loss_h + val_loss_m + val_loss_j + val_loss_w

            vl_ += val_loss.item()
            vm_ += val_loss_m.item()
            vj_ += val_loss_j.item()
            vh_ += val_loss_h.item()
            vw_ += val_loss_w.item()