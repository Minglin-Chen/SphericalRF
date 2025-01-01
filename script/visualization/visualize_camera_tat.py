import os
import os.path as osp

DATA_ROOT = r'..\..\dataset\tanks_and_temples\tanks_and_temples'
OUTPUT_ROOT = r'results\visualize_camera\tanks_and_temples'
COORD_SPEC = 'opengl'
CAMERA_SCALE = 0.1

for s in ['tat_intermediate_M60', 'tat_intermediate_Playground', 'tat_intermediate_Train', 'tat_training_Truck']:
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