import os
import os.path as osp

# configuration
DATA_ROOT = r'..\..\dataset\360_v2'

# run
for s in ['bicycle', 'bonsai', 'counter', 'garden', 'kitchen', 'room', 'stump', 'flowers', 'treehill']:
    command = f'\
        python -m convert_llff2nerf --input "{osp.join(DATA_ROOT, s)}" --output "{osp.join(DATA_ROOT, s)}"'
    print(command)
    os.system(command)