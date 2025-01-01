import os
import os.path as osp

DATA_ROOT = r'..\..\dataset\lf_data\lf_data'
OUTPUT_ROOT = r'results\visualize_camera\lf'
COORD_SPEC = 'opengl'
CAMERA_SCALE = 0.1

for s in ['africa', 'basket', 'ship', 'statue', 'torch']:
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