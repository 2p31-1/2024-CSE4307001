import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

args = argparse.ArgumentParser()
args.add_argument("file1", help="first image file")
args.add_argument("file2", help="second image file")

try:
    # Load left and right images
    left_img = cv2.imread(args.parse_args().file1)
    right_img = cv2.imread(args.parse_args().file2)
except:
    left_img = cv2.imread('data/tsukuba1.ppm', 0)
    right_img = cv2.imread('data/tsukuba2.ppm', 0)

# Display images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Left Image')
plt.imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.title('Right Image')
plt.imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
plt.show()

# Parameters for block matching
window_size = 15
num_disp = 16

# Create Block Matching object
stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=window_size)

# Compute disparity map
disparity = stereo.compute(left_img, right_img)

# Normalize the disparity map for visualization
disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Display disparity map
plt.imshow(disparity, cmap='plasma')
plt.colorbar()
plt.title('Disparity Map')
plt.show()

# Focal length and baseline
f = 500.0  # Replace with actual focal length
B = 1000.0  # Replace with actual baseline distance

# Avoid division by zero
disparity[disparity == 0] = 1

# Compute depth map
depth_map = f * B / disparity

# Normalize depth map for visualization
depth_map = cv2.normalize(depth_map, depth_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
depth_map = np.uint8(depth_map)

# Apply colormap for pseudo-colored image
pseudo_colored_depth = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

# Display depth map
plt.imshow(depth_map, 'gray')
plt.show()
