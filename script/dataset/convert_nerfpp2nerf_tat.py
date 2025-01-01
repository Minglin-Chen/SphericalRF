import os
import os.path as osp

# configuration
DATA_ROOT = r'..\..\dataset\tanks_and_temples\tanks_and_temples'

# run
for s in ['tat_intermediate_M60', 'tat_intermediate_Playground', 'tat_intermediate_Train', 'tat_training_Truck']:
    command = f'\
        python -m convert_nerfpp2nerf --input "{osp.join(DATA_ROOT, s)}" --output "{osp.join(DATA_ROOT, s)}"'
    print(command)
    os.system(command)