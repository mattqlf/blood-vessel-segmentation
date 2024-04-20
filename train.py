import torch as tc
from tqdm import tqdm
import matplotlib.pyplot as plt
from metrics import dice_coef
from config import Config

def train_val(model, train, val, loss_fn, optimizer, scaler, scheduler, epoch, fold):
    model.train()
    avg_train_loss = 0
    avg_train_dice = 0
    time=tqdm(range(len(train)), position=0, leave=True)
    for i, (imgs, labels, _) in enumerate(train):
        imgs = imgs.cuda()
        labels = labels.cuda()
        
        with tc.cuda.amp.autocast():
            preds = model(imgs)
            loss = loss_fn(preds, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()
        
        dice = dice_coef(preds.detach(), labels).item()
        avg_train_loss = (avg_train_loss*i + loss.item())/(i+1)
        avg_train_dice = (avg_train_dice*i + dice)/(i+1)
        time.set_description(f"epoch: {epoch+1} loss: {loss:.4f} dice: {dice:.4f} avg_loss: {avg_train_loss:.4f} avg_dice: {avg_train_dice:.4f}")
        time.update()
    time.close()
    
    model.eval()
    avg_val_loss = 0
    avg_val_dice = 0
    time = tqdm(range(len(val)), position=0, leave=True)
    for i, (imgs, labels, _) in enumerate(val):
        imgs = imgs.cuda()
        labels = labels.cuda()
        
        with tc.no_grad():
            with tc.cuda.amp.autocast():
                preds = model(imgs)
                loss = loss_fn(preds, labels)
                
        if i == len(val) - 1 and fold == Config.folds - 1:
            fig, ax = plt.subplots(1,3, figsize=(20,20))
            preds = tc.sigmoid(preds)
            preds = (preds>0.5).float()
            ax[0].imshow(imgs.cpu()[0].permute(1,2,0))
            ax[1].imshow(labels.cpu()[0].permute(1,2,0))
            ax[2].imshow(preds.cpu().detach()[0].permute(1,2,0))
        
        dice = dice_coef(preds.detach(), labels).item()
        avg_val_loss = (avg_val_loss*i + loss.item())/(i+1)
        avg_val_dice = (avg_val_dice*i + dice)/(i+1)
        time.set_description(f"epoch: {epoch+1} loss: {loss:.4f} dice: {dice:.4f} avg_loss: {avg_val_loss:.4f} avg_dice: {avg_val_dice:.4f}")
        time.update()
    time.close()     
    tc.save(model.state_dict(), f"/kaggle/working/{Config.model}-{Config.encoder}-{epoch}.pth.tar")
    return avg_train_loss, avg_train_dice, avg_val_loss, avg_val_dice

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.early_stop = False

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True