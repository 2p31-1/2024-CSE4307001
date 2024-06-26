import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import threading
import os
from skimage.util import view_as_windows

debug = False

if debug == False:
    args = argparse.ArgumentParser()
    args.add_argument("file1", help="first image file", default='data/tsukuba1.ppm')
    args.add_argument("file2", help="second image file", default='data/tsukuba2.ppm')
    args.add_argument("--window", help="window size for block matching", type=int, default=15)
    args.add_argument("--depthfile", help="(optional) depth file to compare", nargs='?')
    args.add_argument("--output", help="output directory name", default='output')

    # make output dir if not exists

    dir = args.parse_args().output
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Load left and right images
    left_img = cv2.imread(args.parse_args().file1, 0)
    right_img = cv2.imread(args.parse_args().file2, 0)

    try:
        depth_img = cv2.imread(args.parse_args().depthfile, 0)
    except:
        depth_img = None

    window_size = args.parse_args().window
else:
    left_img = cv2.imread('data/tsukuba1.ppm', 0)
    right_img = cv2.imread('data/tsukuba2.ppm', 0)
    depth_img = cv2.imread('data/tsukuba_disp.pgm', 0)
    window_size = 15
    dir = 'tsukuba'


def window(image, x, y):
    return image[y - window_size // 2:y + window_size // 2 + 1, x - window_size // 2:x + window_size // 2 + 1]


def ssd(window1, window2):
    try:
        return np.sum((window1 - window2) ** 2)
    except:
        return float('inf')


windows1 = view_as_windows(left_img, (window_size, window_size))
windows2 = view_as_windows(right_img, (window_size, window_size))

disparity = np.zeros(np.shape(left_img)[0:2])

def row_process(y):
    for x1 in range(np.shape(windows1)[1]):
        minx = 0
        minssd = float('inf')
        for x2 in range(np.shape(windows2)[1]):
            if abs(x1 - x2) > 32:
                continue
            ssd_value = ssd(windows1[y, x1], windows2[y, x2])
            if ssd_value < minssd:
                minssd = ssd_value
                minx = x2
        disparity[y+window_size//2, x1+window_size//2] = abs(x1 - minx)
    print(f"do {window_size}, {y} end")


threads = []
for y in range(np.shape(windows1)[0]):
    t = threading.Thread(target=row_process, args=(y,))
    threads.append(t)

for t in threads:
    t.start()

for t in threads:
    t.join()

# Normalize the disparity map for visualization with grayscale
disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
plt.imshow(disparity, cmap='viridis')
plt.colorbar()

mse = 0
# mse with depth file
if depth_img is not None:
    depth_img = cv2.normalize(depth_img, depth_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    mse = np.mean((depth_img - disparity) ** 2)

# write image on file
plt.savefig(f'{dir}/disparity_{window_size}_{mse}.png')
print(f"saved at {dir}/disparity_{window_size}_{mse}.png")
