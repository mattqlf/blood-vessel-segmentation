import torch as tc
import numpy as np
from random import sample
import matplotlib.pyplot as plt
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import init_load, final_load
from torch.utils.data import DataLoader, Subset
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold
from train import train_val
from prediction import pad_predict
from metrics import dice_coef
from config import Config

k1_path = './data/train/kidney_1_dense'
k2_path = './data/train/kidney_2'
k3_path = './data/train/kidney_3_dense'

x_k1 = init_load(paths = sorted(glob(k1_path + '/images/*')), ref = True)
y_k1 = init_load(paths = sorted(glob(k1_path + '/labels/*')))

x_k2 = init_load(paths = sorted(glob(k2_path + '/images/*')), ref = True)
y_k2 = init_load(paths = sorted(glob(k2_path + '/labels/*')))

x_k3 = init_load(paths = sorted([x.replace('labels', 'images').replace('dense', 'sparse') for x in glob(k3_path + '/labels/*')]), ref = True)                   
y_k3 = init_load(paths = sorted(glob(k3_path + '/labels/*')))

x = [x_k1, x_k2, x_k3] # [x_k1, x_k1_voi, x_k2, x_k3]
y = [y_k1, y_k2, y_k3] # [y_k1, y_k1_voi, y_k2, y_k3]

train_augmentations = A.Compose([
    ToTensorV2(transpose_mask = True),
])

val_augmentations = A.Compose([
    ToTensorV2(transpose_mask = True),
])

test_augmentations = ToTensorV2(transpose_mask = True)

pop_mean = 0.3449155390262604
pop_std = 0.11816135048866272

norm = (pop_mean, pop_std)

dataset = final_load(list_x = x, list_y = y, slices=1, augmentations = train_augmentations, norm = norm, clip=None)
loader = DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = 2, pin_memory = True)

model = smp.Unet(encoder_name = 'resnet50', in_channels = 1, classes = 1, encoder_weights = None)
model = tc.nn.DataParallel(model).cuda()
loss_fn = smp.losses.DiceLoss(mode='binary')
scaler = tc.cuda.amp.GradScaler()
optimizer=tc.optim.AdamW(model.parameters(),lr=Config.lr)
scheduler = tc.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=Config.lr, steps_per_epoch=len(loader), epochs=Config.epochs+1, pct_start=0.1,)

def main():
    tc.backends.cudnn.enabled = True
    tc.backends.cudnn.benchmark = True

    kf = KFold(n_splits=Config.folds, shuffle=True, random_state = 2569)

    epoch_train_loss = []
    epoch_train_dice = []
    epoch_val_loss = []
    epoch_val_dice = []

    for epoch in range(Config.epochs):
        print(f"Epoch {epoch + 1}")
        print("---------------------")
        fold_train_loss = []
        fold_train_dice = []
        fold_val_loss = []
        fold_val_dice = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            print(f"Fold {fold + 1}")
            print("-------")
            train_dataset = final_load(list_x = x, list_y = y, slices=1, augmentations = train_augmentations, norm = norm, clip = Config.clip)
            val_dataset = final_load(list_x = x, list_y = y, slices=1, augmentations = val_augmentations, norm = norm, clip = Config.clip)
            
            # train_dataset = Subset(train_dataset, sample(list(train_idx), 50))
            # val_dataset = Subset(val_dataset, sample(list(val_idx), 5))
            
            train_dataset = Subset(train_dataset, train_idx)
            val_dataset = Subset(val_dataset, val_idx)

            train_loader = DataLoader(train_dataset, batch_size = 1, pin_memory = True, num_workers = 2, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size = 1, pin_memory = True, num_workers = 2, shuffle=False)
            
            train_loss, train_dice, val_loss, val_dice = train_val(model = model, train = train_loader, val = val_loader, loss_fn = loss_fn, optimizer = optimizer, scaler = scaler, scheduler = scheduler, epoch=epoch, fold=fold)
            fold_train_loss.append(train_loss)
            fold_train_dice.append(train_dice)
            fold_val_loss.append(val_loss)
            fold_val_dice.append(val_dice)
            
        epoch_train_loss.append(np.mean(fold_train_loss))
        epoch_train_dice.append(np.mean(fold_train_dice))
        epoch_val_loss.append(np.mean(fold_val_loss))
        epoch_val_dice.append(np.mean(fold_val_dice))
        

    if Config.show_fig:    
        fig, ax = plt.subplots(1,2, figsize=(12,12))
        ax[0].plot(range(1, Config.epochs+1), epoch_train_loss, color='r', label = 'train')
        ax[0].plot(range(1, Config.epochs+1), epoch_val_loss, color = 'b', label = 'val')
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("focal loss")
        ax[0].legend()
        ax[0].set_xticks(range(1,Config.epochs+1))


        ax[1].plot(range(1, Config.epochs+1), epoch_train_dice, color='r', label = 'train')
        ax[1].plot(range(1, Config.epochs+1), epoch_val_dice, color = 'b', label = 'val')
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("dice score")
        ax[1].legend()
        ax[1].set_xticks(range(1,Config.epochs+1))


        plt.tight_layout()
        plt.show()

    if Config.debug:
        test_dataset = final_load(list_x = x, list_y = y, augmentations = test_augmentations, norm = norm, clip = Config.clip, infer=True)
        idx_list = sample(range(len(test_dataset)), 30)

        for idx in idx_list:
            preds = pad_predict(model = model, images = test_dataset[idx][0].unsqueeze(0), divis = 32)
            print(dice_coef(preds, test_dataset[idx][1].unsqueeze(0).cuda()))
            preds = tc.sigmoid(preds)
            preds = (preds>0.5).float()
            
            if Config.show_fig:
                fig, ax = plt.subplots(1,4, figsize=(20,20))
                for a in ax: a.axis('off')
                ax[0].imshow(test_dataset[idx][0].permute(1,2,0), cmap='gray')
                ax[1].imshow(test_dataset[idx][1].permute(1,2,0), cmap='gray')
                ax[2].imshow(preds.squeeze().cpu(), cmap='gray')
                ax[3].imshow(preds.squeeze().cpu(), cmap='gray')
                plt.suptitle(test_dataset[idx][2])
                plt.savefig(fname=f'image {idx}', dpi=350, bbox_inches = 'tight')
        
if __name__ == '__main__':
    main()