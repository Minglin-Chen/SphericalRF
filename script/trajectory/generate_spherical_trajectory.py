import os
import os.path as osp
import math
import numpy as np
import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, help='output filename')
parser.add_argument('--num', type=int, default=100, help='number of images')
parser.add_argument('--radius', type=float, nargs='+', default=[1., 1., 1.], help='sphere radius')
parser.add_argument('--elevation', type=float, default=0., help='elevation of trajectory')
parser.add_argument('--translation', type=float, nargs='+', default=[0., 0., 0.], help='translation')
args = parser.parse_args()


def normalize(v):
    return v / np.linalg.norm(v)


def get_camera(r, elevation, azimuth):
    """
    Args:
        r: camera distance
        elevation: radians
        azimuth: radians
    Returns:
        c2w: camera matrix (world coordinate: right-hand, +z up)
    """
    assert len(r) == 3 or len(r) == 1
    if len(r) == 1:
        r = [r[0], r[0], r[0]]
    assert r[0] > 0 and r[1] > 0 and r[2] > 0

    cam_center = np.array([
        r[0] * math.cos(elevation) * math.cos(azimuth), 
        r[1] * math.cos(elevation) * math.sin(azimuth),
        r[2] * math.sin(elevation)
    ], dtype=np.float32)

    # camera-to-world matrix
    up = np.array([0., 0., 1.], dtype=np.float32)
    lookat  = normalize(- cam_center)
    right   = normalize(np.cross(lookat, up))
    up      = normalize(np.cross(right, lookat))

    c2w = np.stack([right, up, -lookat, cam_center], axis=-1)
    c2w = np.concatenate([c2w, np.array([[0., 0., 0., 1.]], dtype=np.float32)], axis=0)

    return c2w


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
    # pose
    azimuths = np.linspace(0., 360., args.num, endpoint=False)
    elevations = np.array([args.elevation])
    azimuths, elevations = np.meshgrid(azimuths, elevations, indexing="xy")
    azimuths    = [math.radians(v) for v in azimuths.flatten()]
    elevations  = [math.radians(v) for v in elevations.flatten()]
    c2ws = np.array(
        [get_camera(args.radius, elev, azim) for elev, azim in zip(elevations, azimuths)])

    # translation
    assert len(args.translation) == 3
    translation = np.array(args.translation, dtype=np.float32)
    c2ws[:,:3, -1] += translation[None]
    
    # json
    json_dict = to_nerf_poses(c2ws)

    # write
    dirname, filename = osp.dirname(args.filename), osp.basename(args.filename)
    if not osp.exists(dirname): os.makedirs(dirname)
    with open(args.filename, 'w') as f:
        json.dump(json_dict, f, indent=4)