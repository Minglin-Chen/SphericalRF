## Learning Spherical Radiance Field for Efficient 360° Unbounded Novel View Synthesis (TIP 2024)

This is the official repository of SphericalRF, that explores to accelerate nerual radiance fields for 360° unbounded novel view synthesis.

### Requirements

* An **NVIDIA GPU**

* A **C++14** capable compiler. The following choices are recommended:
  
  * **Windows**: Visual Studio 2019 or 2022
  
  * **Linux**: GCC/G++ 8 or higher

* A recent version of **CUDA**. The following choices are recommended:
  
  * Windows: CUDA 11.5 or higher
  
  * Linux: CUDA 10.2 or higher

* **CMake** v3.19 or higher

* **Python** 3.7 or higher

* **PyTorch** 1.11.0

### Compilation

```shell
cmake . -B build
cmake --build build --config RelWithDebInfo -j 16
```

### Dataset Preparation

* **Mip-NeRF-360**
  
  * Step 1: download from [link]([mip-NeRF 360](https://jonbarron.info/mipnerf360/)), then unzip at the *dataset* folder
  
  * Step 2: run `cd script/utils && convert_llff2nerf_mipnerf360.bat`

* **Light-Field** & **Tanks-and-Temples**
  
  * Step 1: download from [link]([Kai-46/nerfplusplus: improves over nerf in 360 capture of unbounded scenes (github.com)](https://github.com/Kai-46/nerfplusplus)), then unzip at the *dataset* folder
  
  * Step 2: run `cd script/utils && convert_nerfpp2nerf_lf.bat && convert_nerfpp2nerf_tat.bat`

* **NeRF-Synthetic**
  
  - download from [link]([NeRF: Neural Radiance Fields (matthewtancik.com)](https://www.matthewtancik.com/nerf)), then unzip at the *dataset* folder

### Run
```bash
# mip-nerf-360
cd experiment/srf_mipnerf360/script
train.bat

# light-field
cd experiment/srf_lightfield/script
train.bat

# tat
cd experiment/srf_tanksandtemples/script
python train.py

# nerf-synthetic
cd experiment/srf_nerfsynthetic/script
python train.py
```


### Acknowledgement

If you find the work useful in your research, please consider citing:

```latex
@article{chen2024learning,
  title={Learning Spherical Radiance Field for Efficient 360° Unbounded Novel View Synthesis},
  author={Chen, Minglin and Wang, Longguang and Lei, Yinjie and Dong, Zilong and Guo, Yulan},
  journal={IEEE Transactions on Image Processing},
  year={2024},
  publisher={IEEE}
}
```
