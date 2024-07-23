# AIDA

### Development Information ###
```
Date Created: 23 July 2024
Developer: Maryam and Amirali
```

### Introduction ###
Investigation of histopathology slides by pathologists is an indispensable component of the routine diagnosis of cancer. Artificial intelligence (AI) has the potential to enhance diagnostic accuracy, improve efficiency, and patient outcomes in clinical pathology. However, variations in tissue preparation, staining protocols, and histopathology slide digitization could result in over-fitting of deep learning models when trained on the data from only one center, thereby underscoring the necessity to generalize deep learning networks for multi-center use. Several techniques, including the use of grayscale images, color normalization techniques, and Adversarial Domain Adaptation (ADA) have been suggested to generalize deep learning algorithms, but there are limitations to their effectiveness and discriminability. Convolutional Neural Networks (CNNs) exhibit higher sensitivity to variations in the amplitude spectrum, whereas humans predominantly rely on phase-related components for object recognition. As such, we propose Adversarial fourIer-based Domain Adaptation (AIDA) which applies the advantages of a Fourier transform in adversarial domain adaptation. We conducted a comprehensive examination of subtype classification tasks in four cancers, incorporating cases from multiple medical centers. Specifically, the datasets included multi-center data for 1113 ovarian cancer cases, 247 pleural cancer cases, 422 bladder cancer cases, and 482 breast cancer cases. Our proposed approach significantly improved performance, achieving superior classification results in the target domain, surpassing the baseline, color augmentation and normalization techniques, and ADA. Furthermore, extensive pathologist reviews suggested that our proposed approach, AIDA, successfully identifies known histotype-specific features. This superior performance highlights AIDA’s potential in addressing generalization challenges in deep learning models for multi-center histopathology datasets.

[Paper (npj Precision Oncology)](https://www.nature.com/articles/s41698-024-00652-4)

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

```
{"chunks": [{"id": 0, "imgs": ["pattern_of_patches/x1_y1.png", "pattern_of_patches/x2_y2.png"]}, {"id": 1, "imgs": ["pattern_of_patches/x3_y3.png", "pattern_of_patches/x4_y4.png"]}, {"id": 2, "imgs": ["pattern_of_patches/x5_y5.png", "pattern_of_patches/x6_y6.png"]}]}
```

### Citation ###

```
@article{asadi2024learning,
  title={Learning generalizable AI models for multi-center histopathology image classification},
  author={Asadi-Aghbolaghi, Maryam and Darbandsari, Amirali and Zhang, Allen and Contreras-Sanz, Alberto and Boschman, Jeffrey and Ahmadvand, Pouya and K{\"o}bel, Martin and Farnell, David and Huntsman, David G and Churg, Andrew and C. Black, Peter and Wang, Gang and Gilks, C. Blake, and Farahani, Hossein and Bashashati, Ali},
  journal={npj Precision Oncology},
  volume={8},
  number={1},
  pages={151},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
