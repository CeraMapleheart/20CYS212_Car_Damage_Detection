# 🚗 Car Damage Detection using Deep Learning

A machine learning project focused on detecting and classifying car damage using deep learning models. This project compares two different architectures for car damage detection, utilizing a dataset of car images from Kaggle. The models were trained and evaluated using TensorFlow and Keras in a Jupyter Notebook environment.

## 📑 Problem Statement

The goal of this project is to accurately detect and classify car damage using a machine learning-based approach. We aim to compare the performance of two different models:
- **Model 1**: MobileNetV2 (Pre-trained)
- **Model 2**: Custom CNN (Convolutional Neural Network)

Both models are assessed based on their accuracy, performance, and resource efficiency.

## 📊 Dataset

- **Source**: [Kaggle Car Damage Dataset](<INSERT LINK HERE>)
- **Size**: 1,920 images of car damages
- **Classes**: Damage / No Damage

## 🏗️ Project Structure

```bash
├── data/                    # Dataset and preprocessed files
├── models/                  # Saved model files
├── notebooks/               # Jupyter notebooks
├── images/                  # Plots, graphs, and result images
├── README.md                # Project documentation
└── requirements.txt         # Required Python libraries
```

## 🛠️ Tools and Technologies

- **Language**: Python
- **Libraries**: TensorFlow, Keras, NumPy, Pandas, Matplotlib, Scikit-learn, OS
- **Jupyter Notebook**: For interactive code and visualizations
- **Transfer Learning**: MobileNetV2 used as a base model in Model 1

## 🧑‍💻 How to Run

1. **Clone the Repository**:
    ```bash
    git clone <repo-url>
    cd car-damage-detection
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download Dataset**: Place the dataset in the `data/` directory. You can find the dataset on [Kaggle](<INSERT LINK HERE>).

4. **Run the Jupyter Notebook**:
    Open the notebook and run the cells to preprocess data, train the models, and visualize results.

## ⚙️ Model Architectures

### Model 1: **MobileNetV2**
- Pre-trained MobileNetV2 architecture
- Used transfer learning to fine-tune the model
- Key steps: Loading base model, attaching custom head, training on car damage dataset

### Model 2: **Custom CNN**
- Convolutional Neural Network (CNN) built from scratch
- Key steps: Convolutional layers for feature extraction, dense layers for classification, with dropout regularization
- Architecture: Conv -> ReLU -> MaxPool -> Dropout -> Fully Connected -> Softmax

## 🚀 Achievements

- **Preprocessing**: Images were preprocessed (resize, normalize, label encoding).
- **Data Augmentation**: Used `ImageDataGenerator` to perform augmentation (rotation, zoom, etc.).
- **Model Training**: Both models were trained and validated on the Kaggle dataset.
- **Results Visualization**: Accuracy and loss plots were generated for training and validation data.
  
## 📈 Results and Performance Comparison

| Metric          | MobileNetV2 | Custom CNN |
|-----------------|-------------|------------|
| Accuracy        | XX%         | XX%        |
| Training Time   | XX minutes  | XX minutes |
| Parameters      | XX Million  | XX Million |
| Inference Speed | XX ms/image | XX ms/image|

- MobileNetV2 showed better **accuracy** with a faster **inference time** due to transfer learning.
- Custom CNN required more time to train but offered more flexibility in terms of tuning the architecture.

## 📊 Visualizations

Training and testing accuracy/loss graphs for both models:

![Model 1 Accuracy Plot](images/model1_accuracy_plot.png)
![Model 2 Accuracy Plot](images/model2_accuracy_plot.png)

## 📂 Flowcharts

Flowcharts outlining the process flow for both models can be found [here](<INSERT LINK HERE>).

## 📜 References

- [Car Damage Detection and Classification](<PAPER 1 LINK>)
- [Car Damage Assessment Based on VGG Models](<PAPER 2 LINK>)
- [A Very Deep Transfer Learning Model for Vehicle](<PAPER 3 LINK>)

## 🛡️ Future Work

- Further fine-tuning of the models for higher accuracy
- Exploring additional architectures like EfficientNet
- Expanding the dataset to include more diverse types of car damage

## 🏁 Conclusion

This project provides a comparative study between MobileNetV2 and a Custom CNN for car damage detection. While both models performed well, the choice of model depends on application requirements like accuracy, speed, and available computational resources.

