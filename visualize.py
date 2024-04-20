import matplotlib.pyplot as plt
from matplotlib import animation, rc
from albumentations.pytorch import ToTensorV2
from dataset import init_load, final_load
from glob import glob
from tqdm import tqdm

def show(dataset, idx):
    image, label, path = dataset[idx]
    print(image.shape, type(image), label.shape, type(label))
    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    ax[0].imshow(image.permute(1,2,0), cmap='gray')
    ax[1].imshow(label.permute(1,2,0), cmap='gray')
    ax[2].imshow(image.permute(1,2,0), cmap='gray')
    ax[2].imshow(label.permute(1,2,0), alpha=0.3)
    for i in range(0,3): ax[i].axis('off')

    plt.subplots_adjust(wspace=0.05)
    plt.title(path)
    plt.show()

def animate(dataset, id_range):
    fig, ax = plt.subplots(1, 3, figsize=(20,20))
    for i in range(0,3): ax[i].axis('off')
    images = []

    for i in tqdm(id_range):
        image, label, _ = dataset[i]

        im1 = ax[0].imshow(image.permute(1,2,0), animated=True, cmap='gray')
        im2 = ax[1].imshow(label.permute(1,2,0), animated=True, cmap='gray')
        im3 = ax[2].imshow(image.permute(1,2,0), animated=True, cmap='gray')
        im4 = ax[2].imshow(label.permute(1,2,0), animated=True, alpha=0.4)
        
        if i == id_range[0]:
            ax[0].imshow(image.permute(1,2,0), cmap='gray')
            ax[1].imshow(label.permute(1,2,0), cmap='gray')
            ax[2].imshow(image.permute(1,2,0), cmap='gray')
            ax[2].imshow(label.permute(1,2,0), alpha=0.4)
        
        images.append([im1, im2, im3, im4])

    ani = animation.ArtistAnimation(fig, images, interval=50, blit=True, repeat_delay=1000)
    ani.save('k3.gif', writer='imagemagick', fps=30)
    plt.close()
    return ani