import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from prediction import pad_predict
from dataset import init_load, final_load
from main import test_augmentations, Config, norm
from glob import glob

def getid(path):
    family = path.split('/')[-3]
    id = path.split('/')[-1].split('.')[0]
    return family + '_' + id

def rle_encode(mask):
    pixel = mask.flatten()
    pixel = np.concatenate([[0], pixel, [0]])
    run = np.where(pixel[1:] != pixel[:-1])[0] + 1
    run[1::2] -= run[::2]
    rle = ' '.join(str(r) for r in run)
    if rle == '':
        rle = '1 0'
    return rle

model = torch.load("./checkpoints/resnet50_baseline.pth.tar")
model.eval()

k5_path = '/kaggle/input/blood-vessel-segmentation/test/kidney_5'
k6_path = '/kaggle/input/blood-vessel-segmentation/test/kidney_6'
x_k5 = init_load(paths = sorted(glob(k5_path + '/labels/*')))
x_k6 = init_load(paths = sorted(glob(k6_path + '/labels/*')))
x = [x_k5, x_k6]

test_dataset = final_load(list_x = x, list_y = x, slices=1, augmentations = test_augmentations, norm = norm, clip=None)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size = 1, num_workers = 2, pin_memory = True)

ids = []
rles = []

for data in tqdm(test_loader):
    try:
        with torch.no_grad():
            pred = model(data[0].cuda())
    except:
        pred = pad_predict(images = data[0].float().cuda(), model=model, divis = 32)
    
    pred = (pred > Config.threshold).float()
    id = getid(''.join(data[2]))
    
    pred = pred.squeeze().to('cpu').numpy()
    rle = rle_encode(pred)
    ids.append(id)
    rles.append(rle)
    
submission = {'id': ids, 'rle': rles}
submission = dict(sorted(submission.items()))
submission = pd.DataFrame(submission)
submission.set_index('id')
submission.to_csv('submission.csv',index=False)