# Weather Project

This project contains scripts and modules for training and testing various machine learning models for image classification. The models are used to classify different weather conditions from images.

## Table of Contents

- [Description](#description)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Requirements](#requirements)
- [Results](#results)
- [License](#license)

## Description

This project provides a framework to train and evaluate different deep learning models on weather image classification. The script allows users to specify the model, training parameters, and dataset location through command-line arguments. The available models include various ResNet architectures and Vision Transformers (ViT).

## Models

The following models are available for use in this project:
- ResNet18
- ResNet34
- ResNet50
- ResNet101
- ResNet152
- ResnetPre (pre-trained ResNet)
- Vit (Vision Transformer)

## Installation

Clone the repository and install the required packages:

```sh
git clone https://github.com/MohammadAsir/weather_project.git
cd weather_project
pip install -r requirements.txt
```
## Usage

To train and evaluate a model, use the following command:

```sh
python main.py --learning_rate <learning_rate> --max_epochs <max_epochs> --batch_size <batch_size> --num_classes <num_classes> --model <model_name>
```
For example, to train a ResNet50 model, you can use:

```sh
python main.py --learning_rate 1.2e-4 --max_epochs 10 --batch_size 32 --num_classes 11 --model ResNet50
```
## Command-Line Arguments

- learning_rate: Learning rate for training (default: 1.2e-4)
- max_epochs: Maximum number of epochs (default: 10)
- batch_size: Batch size for training (default: 32)
- num_classes: Number of classes in the dataset (default: 11)
- model: Model to use (choices: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResnetPre, Vit)

## Data

The dataset should be organized in the following structure:
```sh
data/
  raw/
    train/
      class1/
      class2/
      ...
    val/
      class1/
      class2/
      ...
    test/
      class1/
      class2/
      ...
```
Place your images in the corresponding train, val, and test directories under data/raw.

## Requirements

The required packages can be found in the requirements.txt file. To install them, run:

```sh
pip install -r requirements.txt
```

## Results
Initial testing results:
|         model         | Learning Rate | batch size | epochs till stop(patience = 3) | accuracy |   f1   | test loss | max_epochs |
|:----------------------|:--------------|:-----------|:-------------------------------|:---------|:-------|:----------|:-----------|
| Resnet 50 Pre         | 1.20E-04      | 128        | 5                              | 0.97816  | 0.97779| 0.074185  | 15         |
| Resnet 18             | 1.20E-04      | 128        | 8                              | 0.82823  | 0.82572| 0.564891  | 15         |
| Resnet 34             | 1.20E-04      | 128        | 7                              | 0.73362  | 0.72786| 0.754056  | 15         |
| Resnet 50             | 1.20E-04      | 128        | 8                              | 0.80640  | 0.78795| 0.624946  | 15         |
| Resnet 101            | 1.20E-04      | 128        | 7                              | 0.69723  | 0.69310| 0.943624  | 15         |
| Resnet 152            | 1.20E-04      | 128        | 9                              | 0.77001  | 0.76938| 0.723952  | 15         |
| Vit_b16               | 1.20E-04      | 128        | 4                              | 0.94614  | 0.96468| 0.167579  | 15         |




## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




