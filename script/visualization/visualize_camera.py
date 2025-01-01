import os
import os.path as osp
import numpy as np
from plyfile import PlyData, PlyElement
import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('json', type=str, help='input json path')
parser.add_argument('out_root', type=str, help='output directory')
parser.add_argument('--coord_spec', type=str, default='opengl', 
                        choices=['opengl','opencv'], help='camera coordinate specification')
parser.add_argument('--cam_scale', type=float, default=1.0, help='camera scale')
args = parser.parse_args()

# Coordinate specification
# - OpenGL right hand coordinates: look at -z, up +y, right, +x (default)
# - OpenCV right hand coordinates: look at +z, up -y, right, +x


def xyz_to_ply(xyz, path):
    """
    Args:
        xyz: numpy.ndarray (3,N)
        path: str
    """
    vertex = np.array(
        [(xyz[0,i], xyz[1,i], xyz[2,i]) for i in range(xyz.shape[1])],
        dtype=[('x','f4'), ('y','f4'), ('z','f4')])
    vertex = PlyElement.describe(vertex, 'vertex')
    PlyData([vertex], text=False).write(path)


def xyz_edge_to_ply(xyz, edge, path):
    """
    Args:
        xyz: numpy.ndarray (3,N)
        edge: numpy.ndarray (3,M)
        path: str
    """
    vertex = np.array(
        [(xyz[0,i], xyz[1,i], xyz[2,i], 200, 0, 0) for i in range(xyz.shape[1])],
        dtype=[('x','f4'), ('y','f4'), ('z','f4'), ('red','u1'), ('green','u1'), ('blue','u1')])
    face = np.array(
        [([edge[0,i], edge[1,i], edge[2,i]],) for i in range(edge.shape[1])],
        dtype=[('vertex_indices', 'i4', (3,))])
    e1 = PlyElement.describe(vertex, 'vertex')
    e2 = PlyElement.describe(face, 'face')
    PlyData([e1,e2], text=False).write(path)


def load_cam_model(cam_scale, spec='opencv'):
    """
    Returns:
        xyz: numpy.ndarray (N,3)
        edge: numpy.ndarray (M,3)
    """
    assert spec in ['opengl', 'opencv']

    if spec == 'opengl':
        xyz = np.array([
            [ 0.0, 0.0, 0.0], 
            [-0.5, 0.5,-1.0], 
            [ 0.5, 0.5,-1.0], 
            [ 0.5,-0.5,-1.0], 
            [-0.5,-0.5,-1.0],
            [ 0.0, 0.7,-1.0]])
        edge = np.array([
            [0,1,2],
            [0,2,3],
            [0,3,4],
            [0,4,1],
            [5,1,2]])
    elif spec == 'opencv':
        xyz = np.array([
            [ 0.0, 0.0, 0.0], 
            [-0.5,-0.5, 1.0], 
            [ 0.5,-0.5, 1.0], 
            [ 0.5, 0.5, 1.0], 
            [-0.5, 0.5, 1.0],
            [ 0.0,-0.7, 1.0]])
        edge = np.array([
            [0,1,2],
            [0,2,3],
            [0,3,4],
            [0,4,1],
            [5,1,2]])

    return xyz*cam_scale, edge


def load_cam_matrix(json_path):
    """
    Args:
        json_path: str
    Returns:
        c2ws: numpy.ndarray (N,4,4)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    c2ws = []
    for frame in data['frames']:
        # camera-to-world matrix
        mat = np.array(frame['transform_matrix'], dtype=np.float32)
        c2ws.append(mat)
    # (N,4,4)
    c2ws = np.stack(c2ws)

    return c2ws


def main():
    if not osp.exists(args.out_root):
        os.makedirs(args.out_root)

    json_path = args.json
    assert osp.exists(json_path) and json_path[-5:] == '.json'
    scene = osp.basename(osp.dirname(json_path))
    json_name = osp.basename(json_path)[:-5]

    # load camera model
    cam_xyz, cam_edge = load_cam_model(args.cam_scale, args.coord_spec)

    # laod camera matrix
    c2ws = load_cam_matrix(json_path)

    # transform
    n_cam, n_pt = c2ws.shape[0], cam_xyz.shape[0]
    cam_xyz1 = np.concatenate([cam_xyz, np.ones((n_pt,1))], axis=-1)
    # (n_cam, n_pt, 4)
    cam_xyz1 = np.sum(c2ws[:,None,:,:]*cam_xyz1[None,:,None,:], axis=-1)
    # (n_cam, n_pt, 3)
    xyz = cam_xyz1[...,:3]
    edge = np.stack([cam_edge + i*n_pt for i in range(n_cam)], axis=0)

    # flatten
    xyz, edge = xyz.reshape(-1,3).transpose(), edge.reshape(-1,3).transpose()

    # save
    ply_path = osp.join(args.out_root, f'{json_name}.ply' if scene=='.' else f'{scene}_{json_name}.ply')
    xyz_edge_to_ply(xyz, edge, ply_path)


if __name__=='__main__':
    main()