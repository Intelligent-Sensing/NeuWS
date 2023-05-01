# Neural Wavefront Shaping

Code for the paper "NeuWS: Neural Wavefront Shaping for Guidestar-Free Imaging Through Static and Dynamic Scattering Media" by Brandon Y. Feng, Haiyun Guo, Mingyang Xie, Vivek Boominathan, Manoj K. Sharma, Ashok Veeraraghavan, and Christopher A. Metzler.

## Setup
``` 
!pip install aotools einops 
```

## Run
### Reconstruct Experimental Data
Place the experimental data in the folder `DATA_DIR/SCENE_NAME`. Set the variable `NUM_FRAMES` to the number of frames in the data. Then run the following command:
``` 
python recon_exp_data.py \
        --scene_name SCENE_NAME --num_t NUM_FRAMES --data_dir DATA_DIR \
        --vis_freq -1 --num_epochs 1000 --batch_size 8 --silence_tqdm"
```

### Static Simulation
``` 
python train_static_object.py --img_name Su_27_binary 
```

### Moving Simulation
#### First time
``` 
python train_moving_digits.py --img_name moving_digits --generate_data 
```
#### then
``` 
python train_moving_digits.py --img_name moving_digits 
```

### Run Baseline Experiments for Comparison
``` 
python baseline_IGWS.py --img_name d1-48 
```

