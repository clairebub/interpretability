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
                          [--task {skin,meta,data_statistics}]
```

#### Model training and testing:
- Default is training resnet34
- Can run without any arguments
```bash
python train_model.py [-h]
                      [--gpu_no GPU_NO]
                      [--task {skin,age_approx,anatom_site_general,sex}]
                      [--dataset DATASET]
                      [--ood_dataset {tinyImageNet_resize,LSUN_resize,iSUN,cifar10,cifar100,svhn}]
                      [--model {densenet121,densenet161,densenet169,densenet201,resnet18,resnet34,resnet50,resnet101,resnet152,resnext101_32x8d,vgg13,vgg16}]
                      [--model_customize {base,cosine,ensemble}]
                      [--ood_method {base,odin,cosine}]
                      [--batch_size BATCH_SIZE]
                      [--epochs EPOCHS]
                      [--seed SEED]
                      [--optim {SGD,Adam,RMSprop}]
                      [--learning_rate LEARNING_RATE]
                      [--beta1 BETA1]
                      [--train_augmentation TRAIN_AUGMENTATION]
                      [--test TEST]
                      [--test_augmentation TEST_AUGMENTATION]
                      [--error_analysis ERROR_ANALYSIS]
                      [--generate_result GENERATE_RESULT]
                      [--validation VALIDATION]
                      [--pretrained PRETRAINED]
```

- To train ensemble model, first train two resnet models
(The current code hardcodes the ensemble of resnet101 and resnet152 which has the best performance)

- Check training progress on tensorboard

```bash
tensorboard --logdir label1:path_to_model_1,label2:path_to_model_2
```

## Results (To be updated)

Current results on ISIC 2018 training data (5-fold validation/test):

|Model           |Accuracy                       
|----------------|--------------------
|Densenet121               |83.65%
|Resnet34                  |82.93%
|Resnet18                  |82.48%
|Resnet18 (finetune)       |81.10%
|Resnet34 (finetune)       |80.05%
|Resnet101                 |79.53%
|Resnet50                  |79.28%
|Resnet50 (finetune)       |79.24%
|Densenet121(finetune)     |77.37%
