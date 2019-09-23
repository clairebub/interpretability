# ISIC2019Classification

## Configuration
#### Create a virtual environment
```bash
conda create --name 'env_name'
pip install -r requirements.txt
```

## Usage

#### Data processing:
- Download and unzip data from https://challenge2019.isic-archive.com/
- Put into data folder
```bash
python data_processing.py [-h]
                          [--task {skin,segmentation,meta,data_statistics}]
```
- Step 1: Data preparation for ISIC2019
```bash
python data_processing.py --task skin
```
- Step 2: Data preparation for segmentation (Download and unzip data from ISIC2018)
```bash
python data_processing.py --task segmentation
```

#### Model training and testing:
- Default is training resnet34
- Can run without any arguments
```bash
usage: classifier_entry.py [-h]
                           [--train]
                           [--test]
                           [--gradcam]
                           [--test_segmentation]
                           [--gpu_no GPU_NO]
                           [--task {skin,age_approx,anatom_site_general,sex,general}]
                           [--dataset {isic2019,cifar10,cifar100,fashioniq2019}]
                           [--model {densenet121,densenet161,densenet169,densenet201,resnet18,resnet34,resnet50,resnet101,resnet152,resnext101_32x8d,vgg13,vgg16}]
                           [--model_type {base,cosine,ensemble,ensemble_cosine}]
                           [--ensemble_models ENSEMBLE_MODELS [ENSEMBLE_MODELS ...]]
                           [--batch_size BATCH_SIZE]
                           [--epochs EPOCHS]
                           [--valid_steps VALID_STEPS]
                           [--seed SEED]
                           [--optim {SGD}]
                           [--learning_rate LEARNING_RATE]
                           [--gradcam_conf GRADCAM_CONF]
                           [--gradcam_threshold GRADCAM_THRESHOLD]
                           [--train_augmentation]
                           [--test_augmentation]
                           [--error_analysis]
                           [--generate_result]
                           [--validation]
                           [--pretrained]
                           [--ood_dataset {tinyImageNet_resize,LSUN_resize,iSUN,cifar10,cifar100,svhn}]
                           [--ood_method {Baseline,ODIN,Mahalanobis,Mahalanobis_IPP,DeepMahalanobis,DeepMahalanobis_IPP}]
                           [--data_perturb_magnitude DATA_PERTURB_MAGNITUDE [DATA_PERTURB_MAGNITUDE ...]]
```

- To train ensemble model, first train two resnet models
(The current code hardcodes the ensemble of resnet101 and resnet152 which has the best performance)

- Check training progress on tensorboard

```bash
tensorboard --logdir label1:path_to_model_1,label2:path_to_model_2
```

