import os
import os.path as osp
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm
from glob import glob
import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input', help='input directory')
parser.add_argument('--output', help='output directory')
args = parser.parse_args()


def main():
    if not osp.exists(args.output):
        os.makedirs(args.output)

    # format:
    #   [[fx, 0,  cx, 0],
    #    [0,  fy, cy, 0],
    #    [0,  0,  1,  0],
    #    [0,  0,  0,  1]]
    c2i_path = glob(osp.join(args.input, 'train', 'intrinsics', '*'))[0]
    c2i = np.loadtxt(c2i_path)

    rgb_path = glob(osp.join(args.input, 'train', 'rgb', '*'))[0]
    image = np.array(imageio.imread(rgb_path))
    height, width = image.shape[:2]

    cx, cy = c2i[2] / width, c2i[6] / height
    fx, fy = c2i[0], c2i[5]

    for part in ['train', 'test']:
        rgb_dir = osp.join(args.input, part, 'rgb')
        pose_dir = osp.join(args.input, part, 'pose')

        frames = []
        for rgb_path in tqdm(glob(osp.join(rgb_dir, '*'))):
            basename = osp.basename(rgb_path)
            basename, suffix = basename.split('.')
            assert suffix == 'jpg' or suffix == 'png'

            # get pose
            c2w = np.loadtxt(osp.join(pose_dir, basename+'.txt')).reshape(4,4)
            c2w = c2w @ np.diag([1., -1., -1., 1.])
            c2w = np.array([
                [1., 0., 0., 0.], 
                [0., 0., 1., 0.],
                [0., -1., 0., 0.],
                [0., 0., 0., 1.]]) @ c2w
            
            frames.append({
                'file_path': osp.join(part, "rgb", basename+'.'+suffix),
                'transform_matrix': c2w.tolist()
            })

        json_dict = {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'w': width,
            'h': height,
            'frames': frames
        }

        with open(osp.join(args.output, f'transforms_{part}.json'), 'w') as f:
            json.dump(json_dict, f, indent=4)


if __name__=='__main__':
    main()