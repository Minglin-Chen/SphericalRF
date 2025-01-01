import os
import os.path as osp
import imageio.v2 as imageio
import numpy as np
from glob import glob
from tqdm import tqdm
try:
    from cv2 import cv2
except:
    import cv2


def is_corrupted(image):
    # assert image.shape[2] == 3
    acc_image = np.sum(image[...,:3], axis=2)
    mask = np.zeros_like(acc_image)
    mask[acc_image == 0] = True
    mask[acc_image == 255*3] = True
    corrupted = np.sum(mask) > 0
    return corrupted, (mask * 255).astype(np.uint8)


if __name__=='__main__':

    NERFSYNTHETIC_IMAGE_PATHS = glob(osp.join("../../dataset/nerf_synthetic/", "*", "*", "*.png"))
    MIPNERF360_IMAGE_PATHS = \
        glob(osp.join("../../dataset/360_v2/bicycle/images_4", "*")) + \
        glob(osp.join("../../dataset/360_v2/bonsai/images_2", "*")) + \
        glob(osp.join("../../dataset/360_v2/counter/images_2", "*")) + \
        glob(osp.join("../../dataset/360_v2/flowers/images_4", "*")) + \
        glob(osp.join("../../dataset/360_v2/garden/images_4", "*")) + \
        glob(osp.join("../../dataset/360_v2/kitchen/images_2", "*")) + \
        glob(osp.join("../../dataset/360_v2/room/images_2", "*")) + \
        glob(osp.join("../../dataset/360_v2/stump/images_4", "*")) + \
        glob(osp.join("../../dataset/360_v2/treehill/images_4", "*"))
    LF_IMAGE_PATHS = glob(osp.join("../../dataset/lf_data/lf_data/", "*", "*", "rgb", "*.jpg"))
    TAT_IMAGE_PATHS = glob(osp.join("../../dataset/tanks_and_temples/tanks_and_temples/", "*", "*", "rgb", "*.png"))
    
    for image_path in tqdm(LF_IMAGE_PATHS):
        image = np.array(imageio.imread(image_path))
        corrupted, mask = is_corrupted(image)
        if corrupted:
            print(image_path)
            cv2.imshow("corrupted image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imshow("mask", mask)
            cv2.waitKey()
