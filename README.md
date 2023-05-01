# Neural Wavefront Shaping

Code for the paper "NeuWS: Neural Wavefront Shaping for Guidestar-Free Imaging Through Static and Dynamic Scattering Media" by Brandon Y. Feng, Haiyun Guo, Mingyang Xie, Vivek Boominathan, Manoj K. Sharma, Ashok Veeraraghavan, and Christopher A. Metzler.

## Setup
``` 
!pip install aotools einops 
```

## Run
### Reconstruct Experimental Data
``` 
python recon_exp_data.py \
        --vis_freq -1 --num_t NUM_FRAMES --data_dir DATA_DIR \
        --scene_name SCENE_NAME --num_epochs 1000 --batch_size 8 \
        --max_intensity MAX_RAW_INTENSITY --silence_tqdm"
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
python baseline_Katz.py --img_name d1-48 
```

