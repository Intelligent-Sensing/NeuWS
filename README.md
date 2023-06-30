# Neural Wavefront Shaping
​
Code for the paper "NeuWS: Neural Wavefront Shaping for Guidestar-Free Imaging Through Static and Dynamic Scattering Media" by Brandon Y. Feng, Haiyun Guo, Mingyang Xie, Vivek Boominathan, Manoj K. Sharma, Ashok Veeraraghavan, and Christopher A. Metzler.

https://www.science.org/doi/10.1126/sciadv.adg4671
​
## Setup
Follow these steps to set up the environment:
``` 
conda create -n neuws python=3.9
conda activate neuws
pip install -r requirements.txt
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```
We assume access to a GPU with CUDA 11.3.1 installed/supported.

## Dataset
Please download a NeuWS dataset from https://doi.org/10.5061/dryad.6t1g1jx42. The dataset is also available at https://rice.app.box.com/s/1fbdvj0w7x2xugzs94a02hnagkt7dwvr.

## Reconstruct Experimental Data

Place the experimental data in the folder `DATA_DIR/SCENE_NAME`. Set the variable `NUM_FRAMES` to the number of frames captured in the dataset. 

For scenes containing static scene and static aberration (e.g. Fig. 2 in paper), run the following command:
``` 
python ./recon_exp_data.py \
    --static_phase \
    --num_t NUM_FRAMES --data_dir DATA_DIR/SCENE_NAME/Zernike_SLM_data \
    --scene_name SCENE_NAME --phs_layers 4 --num_epochs 1000 --save_per_frame
```

Example call (will take roughly 4 minutes on an Nvidia 3090 RTX GPU):
``` 
python ./recon_exp_data.py \
    --static_phase \
    --num_t 100 --data_dir ../NeuWS_data/static_objects_static_aberrations/dog_esophagus_0.5diffuser/Zernike_SLM_data  \
    --scene_name dog_esophagus_0.5diffuser --phs_layers 4 --num_epochs 1000 --save_per_frame
```

For scenes containing dynamic scene and dynamic aberration (e.g. Fig. 5 in paper), run the following command:
``` 
python ./recon_exp_data.py \
    --dynamic_scene \
    --num_t NUM_FRAMES --data_dir DATA_DIR/SCENE_NAME/Zernike_SLM_data \
    --scene_name SCENE_NAME --phs_layers 4 --num_epochs 1000 --save_per_frame
```

Example call (will take roughly 17 minutes on an Nvidia 3090 RTX GPU):
``` 
python ./recon_exp_data.py \
    --dynamic_scene \
    --num_t 100 --data_dir ../NeuWS_data/dynamic_objects_dynamic_aberrations/owlStamp_onionSkin/Zernike_SLM_data \
    --scene_name owlStamp_onionSkin --phs_layers 4 --num_epochs 1000 --save_per_frame
```
