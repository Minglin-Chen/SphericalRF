import os
import os.path as osp

# configuration
DATA_ROOT = r'..\..\dataset\lf_data\lf_data'

# run
for s in ['africa', 'basket', 'ship', 'statue', 'torch']:
    command = f'\
        python -m convert_nerfpp2nerf --input "{osp.join(DATA_ROOT, s)}" --output "{osp.join(DATA_ROOT, s)}"'
    print(command)
    os.system(command)