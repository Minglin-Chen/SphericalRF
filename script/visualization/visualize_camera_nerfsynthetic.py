import os
import os.path as osp

DATA_ROOT = r'..\..\dataset\nerf_synthetic'
OUTPUT_ROOT = r'results\visualize_camera\nerf_synthetic'
COORD_SPEC = 'opengl'
CAMERA_SCALE = 0.1

for s in ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']:
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