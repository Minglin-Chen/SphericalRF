import os
import os.path as osp

DATA_ROOT = r'..\..\dataset\360_v2'
OUTPUT_ROOT = r'results\visualize_camera\mipnerf360'
COORD_SPEC = 'opengl'
CAMERA_SCALE = 0.1

for s in ['bicycle', 'bonsai', 'counter', 'garden', 'kitchen', 'room', 'stump', 'flowers', 'treehill']:
    command = f'\
        python -m visualize_camera \
            "{osp.join(DATA_ROOT, s, "transforms_train.json")}" \
            "{OUTPUT_ROOT}" \
            --coord_spec {COORD_SPEC} \
            --cam_scale {CAMERA_SCALE}'
    print(command)
    os.system(command)

    command = f'\
        python -m visualize_camera \
            "{osp.join(DATA_ROOT, s, "transforms_test.json")}" \
            "{OUTPUT_ROOT}" \
            --coord_spec {COORD_SPEC} \
            --cam_scale {CAMERA_SCALE}'
    print(command)
    os.system(command)