# AIDA

### Development Information ###
```
Date Created: 2 Aug 2021
Developer: Maryam and Amirali
```

### About The Project ###
This repo is an implementation of a AIDA


## Installation

```
mkdir AIDA
cd AIDA
git clone git clone https://github.com/AIMLab-UBC/AIDA .
pip install -r requirements.txt
```


### Usage ###
```

usage: run.py [-h] [--experiment_name EXPERIMENT_NAME] [--log_dir LOG_DIR]
              [--chunk_file_location CHUNK_FILE_LOCATION]
              [--chunk_file_location_complete CHUNK_FILE_LOCATION_COMPLETE]
              [--training_chunks TRAINING_CHUNKS [TRAINING_CHUNKS ...]]
              [--validation_chunks VALIDATION_CHUNKS [VALIDATION_CHUNKS ...]]
              [--test_chunks TEST_CHUNKS [TEST_CHUNKS ...]]
              [--patch_pattern PATCH_PATTERN]
              [--subtypes SUBTYPES [SUBTYPES ...]]
              [--num_classes NUM_CLASSES] [--epochs EPOCHS]
              [--num_patch_workers NUM_PATCH_WORKERS]
              [--batch_size BATCH_SIZE] [--eval_batch_size EVAL_BATCH_SIZE]
              [--resize RESIZE] [--fft_enhancer] [--lr LR] [--wd WD] [--seed SEED]
              [--patience PATIENCE] [--lr_patience LR_PATIENCE]
              [--save_method {patch,slide,All}] [--optimizer {Adam,AdamW,SGD}]
              [--use_schedular] [--not_use_weighted_loss]
              [--criteria {overall_auc,overall_acc,balanced_acc,All}]
              [--only_test] [--only_external_test]
              [--external_test_name EXTERNAL_TEST_NAME]
              [--external_chunk_file_location EXTERNAL_CHUNK_FILE_LOCATION]
              [--external_chunks EXTERNAL_CHUNKS [EXTERNAL_CHUNKS ...]]

AIDA

optional arguments:
  -h, --help            show this help message and exit
  --experiment_name EXPERIMENT_NAME
                        experiment name used to name log, model outputs.
  --log_dir LOG_DIR     directory to save the checkpoints and events files.
  --chunk_file_location CHUNK_FILE_LOCATION
                        path to JSON file contains patches address
  --chunk_file_location_complete CHUNK_FILE_LOCATION_COMPLETE
                        path to JSON file contains all the possible patches
                        address
  --training_chunks TRAINING_CHUNKS [TRAINING_CHUNKS ...]
                        space separated number IDs specifying chunks to use
                        for training.
  --validation_chunks VALIDATION_CHUNKS [VALIDATION_CHUNKS ...]
                        space separated number IDs specifying chunks to use
                        for validation.
  --test_chunks TEST_CHUNKS [TEST_CHUNKS ...]
                        space separated number IDs specifying chunks to use
                        for validation.
  --patch_pattern PATCH_PATTERN
                        patterns of the stored patches
  --subtypes SUBTYPES [SUBTYPES ...]
                        space separated words describing subtype=groupping
                        pairs for this study.
  --num_classes NUM_CLASSES
                        number of output classes
  --epochs EPOCHS       number of total epochs to run
  --num_patch_workers NUM_PATCH_WORKERS
                        number of data loading workers
  --batch_size BATCH_SIZE
                        batch size for trianing
  --eval_batch_size EVAL_BATCH_SIZE
                        batch size for validation and testing phase
  --resize RESIZE       If set, resizing the image in augmentation.
  --fft_enhancer        If set, AIDA will run, else ADA.
  --lr LR               initial learning rate
  --wd WD               weight decay
  --seed SEED           seed for initializing training
  --patience PATIENCE   For early stopping
  --lr_patience LR_PATIENCE
                        For early stopping on learning rate
  --save_method {patch,slide,All}
                        Method for saving trained model: 1.patch: based on
                        validation patch accuracy 2. slide: based on
                        validation slide accuracy 3. Both patch and slide
                        accuracy
  --optimizer {Adam,AdamW,SGD}
                        Optimizer for training the model: 1. Adam 2. AdamW 3.
                        SGD
  --use_schedular       Using schedular for decreasig learning rate in a way
                        that if lr_patience has passed, it will be reduced by
                        0.8.
  --not_use_weighted_loss
                        Not using weighted loss.
  --criteria {overall_auc,overall_acc,balanced_acc,All}
                        Criteria for saving the best model: 1.overall_auc: use
                        AUC same as original paper with highest probability
                        2.overall_acc: uses accuracy 3.balanced_acc: balanced
                        accuracy for imbalanced data 4.All: uses all the
                        possible criteriasNOTE: For calculating AUC for
                        multiclasses, OVO is used to mitigate the imbalanced
                        classes.
  --only_test           Only test not train.
  --only_external_test  Only test on the external dataset.
  --external_test_name EXTERNAL_TEST_NAME
                        usefull when testing on multiple external datasets.
  --external_chunk_file_location EXTERNAL_CHUNK_FILE_LOCATION
                        path to JSON file contains external dataset
  --external_chunks EXTERNAL_CHUNKS [EXTERNAL_CHUNKS ...]
                        space separated number IDs specifying chunks to use
                        for testing (default use all the slides).

```

