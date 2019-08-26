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
python data_processing.py
```

#### Model training and testing:
- Default is training densenet121
- Can run without any arguments
```bash
python train_base_model.py [-h] [--dataset DATASET]
                           [--model {densenet121,resnet18,resnet34,resnet50,resnet101,vgg13,vgg16}]
                           [--batch_size BATCH_SIZE]
                           [--epochs EPOCHS]
                           [--seed SEED]
                           [--learning_rate LEARNING_RATE]
                           [--data_augmentation]
                           [--test TEST]
                           [--pretrained PRETRAINED]
```

- Check training progress on tensorboard

```bash
tensorboard --logdir label1:path_to_model_1,label2:path_to_model_2
```

## Results

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
