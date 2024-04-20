'''
ORIGINAL CODE BY HENGCK23 ON KAGGLE
https://www.kaggle.com/code/hengck23/bug-3d-plot-with-pyvista
'''

import cv2
import numpy as np
import pyvista as pv
from tqdm import tqdm

def plot_mesh(kidney, id_range):
    file = [f'./data/train/{kidney}/labels/{i:04d}.tif' for i in id_range]
    mask=[]

    for f in tqdm(file):
        v = cv2.imread(f,cv2.IMREAD_GRAYSCALE)
        mask.append(v)

    mask = np.stack(mask)
    mask = mask/255

    pl = pv.Plotter()
    point1 = np.stack(np.where(mask > 0.1)).T
    pd1 = pv.PolyData(point1)
    mesh1 = pd1.glyph(geom=pv.Cube())
    pl.add_mesh(mesh1, color='blue')
    pl.show()
