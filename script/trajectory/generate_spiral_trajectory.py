import os
import os.path as osp
import math
import numpy as np
import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, help='output filename')
parser.add_argument('--n_image', type=int, default=100, help='number of images')
parser.add_argument('--n_rotation', type=int, default=2, help='number of spiral rotation')
parser.add_argument('--spiral_range', type=float, nargs='+', default=[1., 1., 1.], help='spiral range')
parser.add_argument('--spiral_lookat', type=float, nargs='+', default=[0., 0., -1.], help='lookat position (look at -z)')
parser.add_argument('--translation', type=float, nargs='+', default=[0., 0., 0.], help='translation')
args = parser.parse_args()


def normalize(v):
    return v / np.linalg.norm(v)


def to_nerf_poses(c2w_mats):
    frames = []
    for i, c2w in enumerate(c2w_mats):
        c2w = np.concatenate([c2w[:3,:4], np.array([[0.,0.,0.,1.]])], axis=0)
        frames.append({
            'file_path': f'{i:04d}.png',
            'transform_matrix': c2w.tolist()
        })

    json_dict = {
        'frames': frames
    }

    return json_dict


if __name__=='__main__':
    assert len(args.spiral_range) == 3
    assert len(args.spiral_lookat) == 3
    assert len(args.translation) == 3
    spiral_lookat   = np.array(args.spiral_lookat)
    translation     = np.array(args.translation)

    # pose
    c2ws = []
    for theta in np.linspace(0., 2. * np.pi * args.n_rotation, args.n_image, endpoint=False):
        cam_position = np.array(
            [args.spiral_range[0] * np.cos(theta), 
             args.spiral_range[1] * np.sin(theta), 
             args.spiral_range[2] * np.sin(theta / args.n_rotation * 1.0)]
        )

        cam_up      = np.array([0., 1., 0.])
        cam_lookat  = normalize(spiral_lookat - cam_position)
        cam_right   = normalize(np.cross(cam_lookat, cam_up))
        cam_up      = normalize(np.cross(cam_right, cam_lookat))

        c2w = np.stack([cam_right, cam_up, -cam_lookat, cam_position], axis=-1)
        c2w = np.concatenate([c2w, np.array([[0., 0., 0., 1.]], dtype=np.float32)], axis=0)

        c2ws.append(c2w)

    c2ws = np.stack(c2ws)
    
    # translation
    c2ws[:,:3, -1] += translation[None]
    
    # # json
    json_dict = to_nerf_poses(c2ws)

    # write
    dirname, filename = osp.dirname(args.filename), osp.basename(args.filename)
    if not osp.exists(dirname): os.makedirs(dirname)
    with open(args.filename, 'w') as f:
        json.dump(json_dict, f, indent=4)