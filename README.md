# 🐕 Image Classifier to Identify Dog Breeds

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![CNN](https://img.shields.io/badge/CNN-Image%20Classification-green.svg)](https://en.wikipedia.org/wiki/Convolutional_neural_network)

A deep learning project that uses **Convolutional Neural Networks (CNNs)** to classify images of dogs by breed. This project compares the performance of three pre-trained architectures: **AlexNet**, **ResNet**, and **VGG**.

## 🎯 Project Objective

Build an image classification pipeline that:
1. **Identifies dogs vs. non-dogs** in images
2. **Classifies dog breeds** from 133 possible breeds
3. **Compares CNN architectures** for accuracy and efficiency

## 🏗️ CNN Architectures Compared

| Architecture | Description | Key Features |
|--------------|-------------|--------------|
| **AlexNet** | Pioneer CNN (2012) | 5 conv layers, ReLU activation |
| **ResNet** | Residual Networks | Skip connections, deeper networks |
| **VGG** | Visual Geometry Group | 16-19 layers, 3×3 convolutions |

## 📊 Results Summary

The project evaluates each architecture on:
- **Dog Detection Accuracy:** % of dog images correctly identified as dogs
- **Breed Classification Accuracy:** % of dog breeds correctly classified
- **Non-Dog Detection:** % of non-dog images correctly identified
- **Processing Time:** Runtime efficiency comparison

## 🛠️ Tech Stack
- **Language:** Python 3.8+
- **Framework:** PyTorch with torchvision
- **Pre-trained Models:** ImageNet weights
- **Libraries:** PIL, NumPy, argparse

## 📁 Project Structure
```
├── check_images.py          # Main classification script
├── classifier.py            # CNN classifier wrapper
├── get_input_args.py        # Command-line argument parser
├── get_pet_labels.py        # Extract labels from filenames
├── classify_images.py       # Run classification on images
├── adjust_results4_isadog.py # Adjust results for dog detection
├── calculates_results_stats.py # Calculate statistics
├── print_results.py         # Display results
├── pet_images/              # Sample pet images for testing
└── Results/                 # Output directories
    ├── alexnet_results/
    ├── resnet_results/
    └── vgg_results/
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch torchvision pillow numpy
```

### Running the Classifier
```bash
# Using VGG (recommended for accuracy)
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt

# Using ResNet
python check_images.py --dir pet_images/ --arch resnet --dogfile dognames.txt

# Using AlexNet (fastest)
python check_images.py --dir pet_images/ --arch alexnet --dogfile dognames.txt
```

### Command-Line Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--dir` | Path to pet images folder | `pet_images/` |
| `--arch` | CNN architecture (`vgg`, `resnet`, `alexnet`) | `vgg` |
| `--dogfile` | Text file with dog breed names | `dognames.txt` |

## 📈 Sample Output
```
*** Results Summary ***
Number of Images: 40
Number of Dog Images: 30
Number of "Not-a" Dog Images: 10

CNN Model Architecture: VGG
% Correct Dogs: 100.0%
% Correct Breed: 93.3%
% Correct "Not-a" Dog: 100.0%
```

## 🔑 Key Learnings
- **Transfer Learning:** Leveraging pre-trained ImageNet weights
- **Architecture Comparison:** Understanding CNN design trade-offs
- **Image Preprocessing:** Resizing, normalization for neural networks
- **Model Evaluation:** Accuracy metrics for classification tasks

## 📚 References
- [ImageNet Classification with Deep CNNs (AlexNet)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
- [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385)
- [Very Deep CNNs (VGG)](https://arxiv.org/abs/1409.1556)

## 👤 Author
**Eze Nnamdi Shedrack**  
MS in Engineering Artificial Intelligence  
Carnegie Mellon University Africa

---
*This project demonstrates practical application of deep learning for computer vision tasks.*
