[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/SdXSjEmH)
# EV-HW3: PhysGaussian

This homework is based on the recent CVPR 2024 paper [PhysGaussian](https://github.com/XPandora/PhysGaussian/tree/main), which introduces a novel framework that integrates physical constraints into 3D Gaussian representations for modeling generative dynamics.

You are **not required** to implement training from scratch. Instead, your task is to set up the environment as specified in the official repository and run the simulation scripts to observe and analyze the results.


## Getting the Code from the Official PhysGaussian GitHub Repository
Download the official codebase using the following command:
```
git clone https://github.com/XPandora/PhysGaussian.git
```


## Environment Setup
Navigate to the "PhysGaussian" directory and follow the instructions under the "Python Environment" section in the official README to set up the environment.


## Running the Simulation
Follow the "Quick Start" section and execute the simulation scripts as instructed. Make sure to verify your outputs and understand the role of physics constraints in the generated dynamics.


## Homework Instructions
Please complete Part 1–2 as described in the [Google Slides](https://docs.google.com/presentation/d/13JcQC12pI8Wb9ZuaVV400HVZr9eUeZvf7gB7Le8FRV4/edit?usp=sharing).


# Reference
```bibtex
@inproceedings{xie2024physgaussian,
    title     = {Physgaussian: Physics-integrated 3d gaussians for generative dynamics},
    author    = {Xie, Tianyi and Zong, Zeshun and Qiu, Yuxing and Li, Xuan and Feng, Yutao and Yang, Yin and Jiang, Chenfanfu},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year      = {2024}
}
```

## Enviroment
CUDA 12.6

### install

conda create -n ev_hw3 python=3.9
conda activate ev_hw3

pip install -r requirements.txt
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu121
pip install ninja

pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization/
pip install -e gaussian-splatting/submodules/simple-knn/

#### FFMPEG
sudo apt update
sudo apt install ffmpeg


### Manual Fixes for Compatibility
#### Required to compile with CUDA 12.x toolchain
diff-gaussian-rasterization
In gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h, add the following at the top:
```#include <cstdint>```

simple-knn
In gaussian-splatting/submodules/simple-knn/simple_knn.cu, add the following at the top:
```#include <cfloat>```

## Run Simulation for Jelly and Metal Material (baseline)

### Run Jelly
python gs_simulation.py \
  --model_path ./model/ficus_whitebg-trained/ \
  --output_path ./output/jelly_ficus \
  --config ./config/ficus_config_jelly.json \
  --render_img --compile_video --white_bg

### Run Metal
python gs_simulation.py \
  --model_path ./model/ficus_whitebg-trained/ \
  --output_path ./output/metal_ficus \
  --config ./config/ficus_config_metal.json \
  --render_img --compile_video --white_bg


## Run Simulation for Jelly and Metal for Other Params
python gs_simulation.py \
  --model_path ./model/ficus_whitebg-trained/ \
  --output_path ./output/metal_ficus \
  --config ./config/ficus_config_metal.json \
  --render_img --compile_video --white_bg
