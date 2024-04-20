import torch as tc
import numpy as np
from dataset import init_load, final_load
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob

test_augmentations = ToTensorV2(transpose_mask = True)

k1_path = './data/train/kidney_1_dense'
k2_path = './data/train/kidney_2'
k3_path = './data/train/kidney_3_dense'

x_k1 = init_load(paths = sorted(glob(k1_path + '/images/*')), ref = False)
y_k1 = init_load(paths = sorted(glob(k1_path + '/labels/*')))

x_k2 = init_load(paths = sorted(glob(k2_path + '/images/*')), ref = False)
y_k2 = init_load(paths = sorted(glob(k2_path + '/labels/*')))

x_k3 = init_load(paths = sorted([x.replace('labels', 'images').replace('dense', 'sparse') for x in glob(k3_path + '/labels/*')]), ref = False)                   
y_k3 = init_load(paths = sorted(glob(k3_path + '/labels/*')))

one_x = [x_k1]
one_y = [y_k1]
two_x = [x_k2]
two_y = [y_k2]
three_x = [x_k3]
three_y = [y_k3]

# k1_mean = 0.3529
# k1_std = 0.0421

# k2_mean = 0.5005
# k2_std = 0.0388

# k3_mean = 0.2978
# k3_std = 0.0117

# kvoi_mean = 0.5137
# kvoi_std = 0.0496

# train_norms = [(k1_mean, k1_std), (k2_mean, k2_std), (k3_mean, k3_std)]

one_dataset = final_load(one_x, one_y, slices = 1, augmentations = test_augmentations, norm = None, clip = None, infer=True)
two_dataset = final_load(two_x, two_y, slices=1, augmentations = test_augmentations, norm = None, clip = None, infer=True)
three_dataset = final_load(three_x, three_y, slices = 1, augmentations = test_augmentations, norm = None)

bins = 256

colors = ['r', 'g', 'b']
labels = ['k1', 'k2', 'k3']
datasets = [one_dataset, two_dataset, three_dataset]

for i in range(4):
    dataset = datasets[i]
    volume = tc.cat([dataset[i][0] for i in tqdm(range(0, len(dataset), 4))])
    print(f'mean: {tc.mean(volume)} std: {tc.std(volume)}')
    frq, edges = np.histogram(volume, bins = bins)
    plt.stairs(frq, edges, fill=True, color=colors[i], label = labels[i], alpha=0.3, edgecolor = colors[i])
    del volume, frq, edges, dataset

plt.legend()
plt.show()