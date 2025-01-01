import os
import os.path as osp
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm
import json

from utils.read_write_model import read_model, qvec2rotmat
from utils.read_write_model import read_cameras_binary, read_images_binary
from utils.parse_colmap_camera import parse_cameras
from utils.transform_camera_pose import transform_poses_pca

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input', help='input directory')
parser.add_argument('--output', help='output directory')
args = parser.parse_args()


def to_nerf_poses(c2w_mats, image_paths, fx, fy, cx, cy, w, h, split):
    frames = []
    for c2w, path in zip(c2w_mats, image_paths):
        path_dirname = osp.basename(osp.dirname(path))
        path_basename = osp.basename(path)
        c2w = np.concatenate([c2w[:3,:4], np.array([[0.,0.,0.,1.]])], axis=0)
        frames.append({
            'file_path': osp.join(path_dirname, path_basename),
            'transform_matrix': c2w.tolist()
        })

    json_dict = {
        'fx': fx,
        'fy': fy,
        'cx': cx / w,
        'cy': cy / h,
        'w' : int(w),
        'h' : int(h),
        'frames': frames
    }

    return json_dict


def main():
    # print(f'Input: {args.input}\nOutput: {args.output}')

    scene = osp.basename(args.input)
    # auto determine the factor
    # compatible to the experiment of MipNeRF 360 
    factor = None
    if scene in ['bicycle', 'garden', 'stump', 'flowers', 'treehill']:
        factor = 4
    elif scene in ['bonsai', 'counter', 'kitchen', 'room']:
        factor = 2
    else:
        raise NotImplementedError
    assert factor is not None
    image_dir = osp.join(args.input, 'images' if factor==1 else f'images_{factor}')

    # load from colmap
    colmap_images   = read_images_binary(os.path.join(args.input, 'sparse', '0', 'images.bin'))
    colmap_cameras  = read_cameras_binary(os.path.join(args.input, 'sparse', '0', 'cameras.bin'))

    # pasring
    image_paths, c2w_mats = [], []
    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    for colmap_image in tqdm(colmap_images.values()):
        # 1. paths
        image_paths.append(os.path.join(image_dir, colmap_image.name))

        # 2. extrinsic matrix (camera-to-world) under the OpenCV specification
        R = qvec2rotmat(-colmap_image.qvec)
        t = colmap_image.tvec.reshape((3,1))
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(w2c)
        c2w_mats.append(c2w[:3,:4])

    # (N,3,4)
    c2w_mats = np.stack(c2w_mats)

    # Switch from OpenCV to OpenGL specification
    c2w_mats = c2w_mats @ np.diag([1, -1, -1, 1])

    # sort by alphabetical order
    inds = np.argsort([os.path.basename(p) for p in image_paths])
    c2w_mats = c2w_mats[inds]
    image_paths = [ image_paths[i] for i in inds ]

    # intrinsic parameters
    _, _, fx, fy, cx, cy, _, _, _, _ = parse_cameras(colmap_cameras)
    fx /= float(factor)
    fy /= float(factor)
    cx /= float(factor)
    cy /= float(factor)

    # Rotate/scale poses to align ground with xy plane and fit to unit cube.
    c2w_mats, _ = transform_poses_pca(c2w_mats)

    # split the dataset
    all_indices = np.arange(len(image_paths))
    train_indices   = all_indices % 8 != 0
    test_indices    = all_indices % 8 == 0
    train_image_paths   = [ image_paths[i] for i, b in enumerate(train_indices) if b ]
    test_image_paths    = [ image_paths[i] for i, b in enumerate(test_indices) if b ]
    train_c2w_mats  = c2w_mats[train_indices]
    test_c2w_mats   = c2w_mats[test_indices]

    # image shape
    h, w = imageio.imread(train_image_paths[0]).shape[:2]

    # covert to nerf-style
    train_dict  = to_nerf_poses(train_c2w_mats, train_image_paths, fx, fy, cx, cy, w, h, 'train')
    test_dict   = to_nerf_poses(test_c2w_mats, test_image_paths, fx, fy, cx, cy, w, h, 'test')

    # write - transforms
    with open(osp.join(args.output, 'transforms_train.json'), 'w') as f:
        json.dump(train_dict, f, indent=4)
    with open(osp.join(args.output, 'transforms_test.json'), 'w') as f:
        json.dump(test_dict, f, indent=4)

    # # copy - images
    # train_image_dir = osp.join(args.output, 'train')
    # if not osp.exists(train_image_dir): os.makedirs(train_image_dir)
    # for path in tqdm(train_image_paths):
    #     imageio.imsave(osp.join(train_image_dir, osp.basename(path)), imageio.imread(path))
    # test_image_dir = osp.join(args.output, 'test')
    # if not osp.exists(test_image_dir): os.makedirs(test_image_dir)
    # for path in tqdm(test_image_paths):
    #     imageio.imsave(osp.join(test_image_dir, osp.basename(path)), imageio.imread(path))


if __name__=='__main__':
    main()