import os
import os.path as osp
import platform


# helpers
def add_pythonpath(p):

    if platform.system() == 'Windows':
        SEP = ';'
    elif platform.system() == 'Linux':
        SEP = ':'
    else:
        raise NotImplementedError
    
    pythonpath = os.environ.get('PYTHONPATH')
    pythonpath = p + SEP + pythonpath if pythonpath else p
    os.environ['PYTHONPATH'] = pythonpath


# configuration
NGPPATH = r'..\..\..\build'
add_pythonpath(r'..\python')
add_pythonpath(NGPPATH)

DATA_ROOT = r'..\..\..\dataset\nerf_synthetic'
RESULT_ROOT = r'..\checkpoint'
N_STEPS = 50000


# run
for s in ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']:

    network = osp.join("..", "config", f"srf_{s}.json")
    save_snapshot = osp.join(RESULT_ROOT, s, f"{s}_ckpt.msgpack")
    if not osp.exists(osp.dirname(save_snapshot)): 
        os.makedirs(osp.dirname(save_snapshot))

    command = f'\
        python -m run \
            --scene {osp.join(DATA_ROOT, s, "transforms_train.json")} \
            --scene_offset 0.5 0.5 0.5 \
            --test_transforms {osp.join(DATA_ROOT, s, "transforms_test.json")} \
            --network {network} \
            --save_snapshot {save_snapshot} \
            --n_steps {N_STEPS}'
    print(command)
    os.system(command)