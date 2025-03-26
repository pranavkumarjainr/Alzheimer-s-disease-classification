## Alzheimer's Disease Classification using Deep Learning

This notebook demonstrates the use of various deep learning models to classify Alzheimer's disease stages based on brain MRI images.

### Models Evaluated:
1. Custom CNN
2. ResNet50
3. VGG16
4. InceptionV3
5. InceptionResNetV2

### Dataset:
The dataset consists of brain MRI images categorized into 4 classes representing different stages of Alzheimer's disease.

### Key Features:
- Data augmentation using ImageDataGenerator
- Transfer learning with pre-trained models
- Fine-tuning of model architectures
- Learning rate scheduling
- Performance visualization

### Results:
| Model             | Loss     | Accuracy |
|-------------------|----------|----------|
| CNN               | 1.033049 | 0.750000 |
| ResNet50          | 1.004276 | 0.779711 |
| VGG16             | 0.942965 | 0.791243 |
| InceptionV3       | 1.012558 | 0.774238 |
| InceptionResNetV2 | 0.970840 | 0.785379 |

### Conclusion:
The VGG16 model achieved the highest accuracy of 79.12% in classifying Alzheimer's disease stages.

### Requirements:
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Scikit-image

### Usage:
1. Mount Google Drive (if using Google Colab)
2. Unzip the dataset
3. Run the notebook cells sequentially to train and evaluate each model

Note: Adjust hyperparameters and model architectures as needed for further experimentation.
