# import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import threading
import os

debug = True


if debug == False:
    args = argparse.ArgumentParser()
    args.add_argument("file1", help="first image file", default='data/tsukuba1.ppm')
    args.add_argument("file2", help="second image file", default='data/tsukuba2.ppm')
    args.add_argument("--window", help="window size for block matching", type=int, default=15)
    args.add_argument("--depthfile", help="(optional) depth file to compare", nargs='?')
    args.add_argument("--output", help="output directory name", default='output')

    #make output dir if not exists

    dir = args.parse_args().output
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Load left and right images
    left_img = plt.imread(args.parse_args().file1)
    right_img = plt.imread(args.parse_args().file2)

    try:
        depth_img = plt.imread(args.parse_args().depthfile)
    except:
        depth_img = None

    window_size = args.parse_args().window
else:
    left_img = plt.imread('data/tsukuba1.ppm')
    right_img = plt.imread('data/tsukuba2.ppm')
    depth_img = plt.imread
    window_size = 15

def window(image, x, y):
    return image[y - window_size // 2:y + window_size // 2 + 1, x - window_size // 2:x + window_size // 2 + 1]


def ssd(window1, window2):
    return np.sum((window1 - window2) ** 2)


disparity = np.zeros(np.shape(left_img))


def row_process(y):
    for x1 in range(np.shape(left_img)[1]):
        if x1 - window_size // 2 < 0 or x1 + window_size // 2 >= np.shape(left_img)[
            1] or y - window_size // 2 < 0 or y + window_size // 2 >= np.shape(left_img)[0]:
            continue
        left_window = window(left_img, x1, y)
        minssd = float('inf')
        minx = -1
        for x2 in range(np.shape(right_img)[1]):
            if x2 - window_size // 2 < 0 or x2 + window_size // 2 >= np.shape(right_img)[1]:
                continue
            right_window = window(right_img, x2, y)
            ssd_value = ssd(left_window, right_window)
            if ssd_value < minssd:
                minssd = ssd_value
                minx = x2
        disparity[y, x1] = abs(x1 - minx)
    print(f"do {window_size}, {y} end")


threads = []
for y in range(np.shape(left_img)[0]):
    t = threading.Thread(target=row_process, args=(y,))
    threads.append(t)

for t in threads:
    t.start()

for t in threads:
    t.join()

# Normalize the disparity map for visualization with grayscale
# disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
plt.imshow(disparity, cmap='gray')
plt.colorbar()
plt.show()

mse = 0
# mse with depth file
if depth_img is not None:
    # depth_img = cv2.normalize(depth_img, depth_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    mse = np.mean((depth_img - disparity) ** 2)

#write image on file
plt.savefig(f'{dir}/disparity_{window_size}_{mse}.png')
print(f"saved at {dir}/disparity_{window_size}_{mse}.png")