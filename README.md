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

# Detailed Image Classification Process Flowchart

```mermaid
flowchart TD
    A[Start] --> B[Import Libraries]
    B --> C[Define Paths and Categories]
    C --> D[Load and Preprocess Images]
    D --> E[One-Hot Encode Labels]
    E --> F[Split Data into Training and Testing Sets]
    F --> G[Data Augmentation Setup]
    G --> H[Build MobileNetV2 Model]
    H --> I[Freeze Base Model Layers]
    I --> J[Compile the Model]
    J --> K[Train the Model]
    K --> L[Evaluate the Model]
    L --> M[Save the Model]
    M --> N[Plot Training Results]
    N --> O[Load and Visualize Model Layers]
    O --> P[Make Predictions]
    P --> Q[End]
    
    B --> R[Libraries Used]
    R --> S[Numpy]
    R --> T[Pandas]
    R --> U[TensorFlow]
    R --> V[Scikit-Learn]
    R --> W[OpenCV]
    R --> X[Matplotlib]
    
    S --> B
    T --> B
    U --> B
    V --> B
    W --> B
    X --> B

    C --> C1[Specify Dataset Path]
    C --> C2[Define Categories: 00-damage, 01-whole]

    D --> D1[Load Images from Training Directory]
    D --> D2[Resize Images to (224, 224)]
    D --> D3[Convert Images to Arrays]
    D --> D4[Preprocess Images with MobileNetV2 Preprocessing]

    E --> E1[Convert Labels to Binary Format]
    E --> E2[Apply One-Hot Encoding]

    F --> F1[Split Data: 80% Training, 20% Testing]
    
    G --> G1[Set Up ImageDataGenerator for Augmentation]
    G --> G2[Define Augmentation Parameters: Rotation, Zoom, etc.]

    H --> H1[Load MobileNetV2 Base Model with Pretrained Weights]
    H --> H2[Add Custom Head: MaxPooling2D, Flatten, Dense Layers]

    I --> I1[Freeze Layers of the Base Model]

    J --> J1[Compile Model with Adam Optimizer]
    J --> J2[Set Loss Function: Binary Crossentropy]
    J --> J3[Define Metrics: Accuracy]

    K --> K1[Train Model Using Augmented Data]
    K --> K2[Use TensorBoard for Visualization]

    L --> L1[Evaluate Model on Testing Set]
    L --> L2[Generate Classification Report]

    M --> M1[Save Trained Model to Disk]

    N --> N1[Plot Training Loss and Accuracy]
    N --> N2[Save Plots as Images]

    O --> O1[Load Pre-trained Model]
    O --> O2[Visualize Activations of Intermediate Layers]
    
    P --> P1[Load Test Images]
    P --> P2[Preprocess Test Images]
    P --> P3[Make Predictions on Test Images]
    P --> P4[Display Prediction Results]


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

