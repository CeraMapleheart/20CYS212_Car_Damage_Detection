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

| Metric          | MobileNetV2 | 
|-----------------|-------------|
| Accuracy        | 78.7%       |
| Parameters      | 2,422,210   | 


- MobileNetV2 showed better **accuracy** with a faster **inference time** due to transfer learning.
- Custom CNN required more time to train but offered more flexibility in terms of tuning the architecture.

## 📊 Visualizations

Training and testing accuracy/loss graphs for both models:

![Accuracy Plot]([images/model1_accuracy_plot.png](https://github.com/user-attachments/assets/bfa126bd-56e7-4503-97b0-be387b6c7feb))


## 📂 Flowcharts

# Detailed Flowchart for Car Damage Detection using MobileNetV2

## 1. Data Loading and Preparation
- **Process**: Load and preprocess images from directories.
  - **Libraries**: 
    - `os`: Directory traversal and file path handling.
    - `numpy`: Numerical operations for data manipulation.
    - `pandas`: Data handling (not specifically used in this case).
    - `tensorflow.keras.preprocessing.image`: For image processing.
  - **Sample Size**: 460 images (230 for "00-damage", 230 for "01-whole").
  - **Purpose**: Gather images, preprocess, and prepare for model input.
  - **Output**: `data` and `labels` arrays.

## 2. Data Preprocessing and Augmentation
- **Process**: Preprocess and augment the data for training and validation.
  - **Libraries**: 
    - `ImageDataGenerator`: Perform real-time data augmentation.
    - `tensorflow.keras.applications.mobilenet_v2`: Preprocess images for MobileNetV2.
    - `sklearn.preprocessing.LabelBinarizer`: One-hot encode labels.
  - **Augmentations**: Rotation, zoom, width/height shifts, flips, etc.
  - **Purpose**: Improve model generalization by generating augmented samples.
  - **Output**: Augmented training data.

## 3. Model Creation and Compilation
- **Process**: Load MobileNetV2, freeze base layers, and add custom classification layers.
  - **Libraries**:
    - `tensorflow.keras.applications.MobileNetV2`: Use pre-trained MobileNetV2 as the base model.
    - `tensorflow.keras.layers`: Add custom fully connected layers on top.
    - `tensorflow.keras.models`: Define and compile the model.
    - `Adam Optimizer`: Set exponential learning rate decay schedule.
  - **Purpose**: Create a model suitable for binary classification.
  - **Output**: Compiled model.

## 4. Model Training
- **Process**: Train the model on augmented training data.
  - **Libraries**:
    - `tensorflow.keras.Model.fit`: Train the model.
    - `TensorBoard`: Log performance metrics.
  - **Training Parameters**: 
    - Learning Rate: `INIT_LR = 1e-5`
    - Epochs: `100`
    - Batch Size: `64`
  - **Purpose**: Train the head of the MobileNetV2 model.
  - **Output**: Trained model and training logs.

## 5. Model Evaluation
- **Process**: Evaluate model performance using testing data.
  - **Libraries**:
    - `tensorflow.keras.Model.predict`: Predict on test data.
    - `sklearn.metrics.classification_report`: Generate classification report.
  - **Evaluation Metrics**: Precision, recall, F1-score, accuracy.
  - **Output**: Performance report with metrics.

## 6. Model Saving
- **Process**: Save the trained model for future inference.
  - **Libraries**: 
    - `tensorflow.keras.models.save_model`: Save model in `.h5` format.
  - **Purpose**: Save the trained model to disk.
  - **Output**: Saved model file (`MobileNet_Car_detection_epoch100_.model`).

## 7. Visualization of Model Layers
- **Process**: Visualize model layer activations for given input images.
  - **Libraries**:
    - `keras.models`: Load and manipulate the saved model.
    - `matplotlib.pyplot`: Display the visualizations.
  - **Purpose**: Analyze how the model processes images at different layers.
  - **Output**: Feature maps of intermediate layers.

## 8. Additional Features: Visualization and TensorBoard
- **Process**: Visualize training loss/accuracy and use TensorBoard for performance monitoring.
  - **Libraries**: 
    - `matplotlib.pyplot`: Plot graphs.
    - `TensorBoard`: Log and visualize performance during training.
  - **Output**: Plots for training loss and accuracy, and TensorBoard logs.

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

