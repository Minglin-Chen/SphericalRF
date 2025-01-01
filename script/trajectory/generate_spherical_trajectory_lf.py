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
add_pythonpath(r'..\visualization')

OUTPUT_ROOT = r'results\generate_spherical_trajectory\lf'