# EV-HW3: PhysGaussian

## Enviroment

CUDA 12.6

### install

```
conda create -n ev_hw3 python=3.9
conda activate ev_hw3

pip install -r requirements.txt
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu121
pip install ninja

pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization/
pip install -e gaussian-splatting/submodules/simple-knn/
```

#### FFMPEG

```
sudo apt update
sudo apt install ffmpeg
```

### Manual Fixes for Compatibility

#### Required to compile with CUDA 12.x toolchain

diff-gaussian-rasterization
In gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h, add the following at the top:
`#include <cstdint>`

simple-knn
In gaussian-splatting/submodules/simple-knn/simple_knn.cu, add the following at the top:
`#include <cfloat>`

## Part1: Run Simulation for Jelly and Metal Material (baseline)

### Run Jelly (Baseline)

```
python gs_simulation.py \
  --model_path ./model/ficus_whitebg-trained/ \
  --output_path ./output/jelly_ficus \
  --config ./config/ficus_config_jelly.json \
  --render_img --compile_video --white_bg
```

### Run Metal (Baseline)

```
python gs_simulation.py \
  --model_path ./model/ficus_whitebg-trained/ \
  --output_path ./output/metal_ficus \
  --config ./config/ficus_config_metal.json \
  --render_img --compile_video --white_bg
```

### 🎥 Part 1 Simulation Videos

Baseline simulations were generated for the following materials:

- **Jelly (ficus)**: [🔗](https://www.youtube.com/watch?v=OCFRxT-7DZ4)
- **Metal (ficus)**: [🔗](https://www.youtube.com/watch?v=KiVUmkzatCI)

## Part 2: Parameter Adjustment (e.g., soften=0 for Jelly)

### Run Jelly (soften=0)

```
python gs_simulation.py \
  --model_path ./model/ficus_whitebg-trained/ \
  --output_path ./output/jelly_ficus_soften0 \
  --config ./config/ficus_config_jelly_soften0.json \
  --render_img --compile_video --white_bg
```

### PSNR Evaluation (e.g., soften=0 vs baseline)

```
python calc_psnr.py --ref_dir output/jelly_ficus_base --test_dir output/jelly_ficus_soften0
```

### 📊 PSNR Results Summary

| Exp # | Parameter Setting          | Avg. PSNR | Notes                                               | YouTube Link                                      |
| ----- | -------------------------- | --------- | --------------------------------------------------- | ------------------------------------------------- |
| 1     | softening=0.1 (default)    | -         | Baseline configuration                              | [🔗](https://www.youtube.com/watch?v=OCFRxT-7DZ4) |
| 2     | softening=0.0              | 76.17     | Nearly identical to baseline, slightly more elastic | [🔗](https://www.youtube.com/watch?v=yfauHmNfbVs) |
| 3     | softening=0.5              | 76.49     | No perceptible visual change                        | [🔗](https://www.youtube.com/watch?v=v2fcGoEq9v4) |
| 4     | softening=1.0              | 76.54     | Same as above, softening has no effect on jelly     | [🔗](https://www.youtube.com/watch?v=PTuKgXT3ovQ) |
| 5     | substep_dt=5e-5            | 22.08     | Ficus appears stiffer, less bounce                  | [🔗](https://www.youtube.com/watch?v=-FYsfx3668o) |
| 6     | substep_dt=5e-4            | -         | Simulation failed (OOM)                             | -                                                 |
| 7     | grid_lim=1.0               | -         | Simulation failed (OOM)                             | -                                                 |
| 8     | grid_lim=3.0               | 21.92     | Ficus less elastic, slightly damped motion          | [🔗](https://www.youtube.com/watch?v=kQ1PoAQ8jns) |
| 9     | n_grid=25                  | 21.96     | Slightly damped motion, low-res grid artifacts rare | [🔗](https://www.youtube.com/watch?v=JM5jRvQttJI) |
| 10    | n_grid=75                  | -         | Simulation failed (OOM)                             | -                                                 |
| 11    | substep_dt=5e-5, n_grid=75 | 21.97     | Ficus motion appears slightly stiffened             | [🔗](https://www.youtube.com/watch?v=vROaJ2WtpmA) |

FULL EXPERIMENTS PLAYLIST : [▶️ Playlist](https://www.youtube.com/playlist?list=PLKdM4OqX00sFmwXJuu9Fd5ESgvOluWfFy)

## 🔍 Part 2 Analysis: Ablation Study & Observations

Across different parameter variations (e.g., `substep_dt`, `grid_lim`, `n_grid`), we observed that:

- Smaller `substep_dt` led to significantly damped motion in jelly, with lower PSNR values (~22), indicating stiffer dynamics.
- Changing `softening` had almost no visible effect on jelly, confirming that this parameter is likely ineffective for elastic materials like jelly.
- Increasing `grid_lim` allowed the object to spread out more during motion, also lowering PSNR but with subtle visual difference.
- Adjusting `n_grid` to a coarser resolution preserved simulation stability, while higher values led to OOM.

Overall, `substep_dt` and `grid_lim` had the most noticeable influence on jelly deformation. These results are consistent with the expected role of time-step resolution and simulation boundary size.
