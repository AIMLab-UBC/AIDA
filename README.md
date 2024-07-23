# AIDA

### Development Information ###
```
Date Created: 23 July 2024
Developer: Maryam and Amirali
```

### About The Project ###
This repo is an implementation of AIDA


## Installation

```
mkdir AIDA
cd AIDA
git clone git clone https://github.com/AIMLab-UBC/AIDA .
pip install -r requirements.txt
```


### Usage ###
```
python3 run.py \
--experiment_name exp_name  \
--log_dir  path_to_dir \
--chunk_file_location path_to_json_file_source \
--patch_pattern pattern_of_patches \
--subtypes label0=0 label1=1 label2=2 ... labeln=n \
--num_classes n \
--epochs 20 \
--num_patch_worker 6 \
--batch_size 32 \
--eval_batch_size 64 \
--fft_enhancer \
--lr 0.0001 \
--seed 7 \
--external_test_name target_name \
--external_chunk_file_location path_to_json_file_target

```
 
### Sample JSON ###
Each file consists of three IDs (0, 1, and 2), with 0 representing training data, 1 representing validation data, and 2 representing test data.

{"chunks": [{"id": 0, "imgs": ["pattern_of_patches/x1_y1.png", "pattern_of_patches/x2_y2.png"]}, {"id": 1, "imgs": ["pattern_of_patches/x3_y3.png", "pattern_of_patches/x4_y4.png"]}, {"id": 2, "imgs": ["pattern_of_patches/x5_y5.png", "pattern_of_patches/x6_y6.png"]}]}
