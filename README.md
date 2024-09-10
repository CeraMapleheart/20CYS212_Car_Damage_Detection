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
+----------------------------------------------------------+
|                       Start                              |
+----------------------------------------------------------+
                        |
                        v
+----------------------------------------------------------+
|  Import Necessary Libraries                              |
|  Tools: TensorFlow, Keras, NumPy, Pandas                 |
|  Why: TensorFlow/Keras for model building,               |
|       NumPy for array operations, Pandas for dataframes  |
+----------------------------------------------------------+
                        |
                        v
+----------------------------------------------------------+
|     Define Image Paths and Label Categories              |
|  Tools: OS (for file handling), Pandas                   |
|  Why: OS helps manage the directory structure,           |
|       Pandas assists with organizing the data labels     |
+----------------------------------------------------------+
                        |
                        v
+----------------------------------------------------------+
| Check if Paths and Categories Exist                      |
|  Tools: OS                                               |
|  Why: To verify if the correct directories and files are |
|       available before loading data                      |
+----------------------------------------------------------+
                        |
                        v
+----------------------------------------------------------+
|             Load Images from Directory                   |
|  Tools: Keras (image_dataset_from_directory)             |
|  Why: Efficiently loads and labels images while scaling  |
|       them into a usable format                          |
+----------------------------------------------------------+
                        |
                        v
+----------------------------------------------------------+
|              Preprocess Images                           |
|  Tools: TensorFlow (tf.image), Keras                     |
|  Why: Resize images to 224x224 (for consistency) and     |
|       normalize pixel values for faster model convergence|
+----------------------------------------------------------+
|  Resize Images (224x224)                                 |
|  Normalize Pixel Values (0 to 1 scale)                   |
+----------------------------------------------------------+
                        |
                        v
+----------------------------------------------------------+
|             One-Hot Encode Labels                        |
|  Tools: TensorFlow (tf.keras.utils.to_categorical)       |
|  Why: Converts categorical labels into a binary vector   |
|       form required for classification tasks             |
+----------------------------------------------------------+
                        |
                        v
+----------------------------------------------------------+
|  Split Data into Training and Testing Sets               |
|  Tools: scikit-learn (train_test_split)                  |
|  Why: To create a reliable 80-20 training-test split,    |
|       ensuring the model doesn’t learn the test data     |
+----------------------------------------------------------+
|  80% Training, 20% Testing                               |
|  Shuffle and ensure no data leakage                      |
+----------------------------------------------------------+
                        |
                        v
+----------------------------------------------------------+
|          Apply Data Augmentation (Optional)              |
|  Tools: TensorFlow (ImageDataGenerator)                  |
|  Why: Enhances training data by applying random          |
|       transformations (flips, rotations, zooms) to avoid|
|       overfitting                                        |
+----------------------------------------------------------+
                        |
                        v
+----------------------------------------------------------+
|         Load Pre-Trained MobileNetV2 Model               |
|  Tools: Keras (keras.applications.MobileNetV2)           |
|  Why: MobileNetV2 is a lightweight pre-trained model that|
|       is efficient for image classification tasks        |
+----------------------------------------------------------+
| (exclude top layers - include_top=False)                 |
+----------------------------------------------------------+
                        |
                        v
+----------------------------------------------------------+
|           Freeze Base Model Layers                       |
|  Tools: Keras (model.trainable = False)                  |
|  Why: Prevents pre-trained layers from being modified,   |
|       ensuring their learned weights are preserved       |
+----------------------------------------------------------+
                        |
                        v
+----------------------------------------------------------+
|             Add Custom Dense Layers                      |
|  Tools: Keras (Sequential, Dense, Flatten, Softmax)      |
|  Why: To adapt the model for your specific classification|
|       task by adding custom layers on top of MobileNetV2 |
+----------------------------------------------------------+
|  Flatten layer + Dense Layer + Softmax Output            |
+----------------------------------------------------------+
                        |
                        v
+----------------------------------------------------------+
|                 Compile the Model                        |
|  Tools: Keras (model.compile)                            |
|  Why: Set the loss function, optimizer, and metrics to   |
|       guide how the model trains                         |
+----------------------------------------------------------+
|  Loss: categorical_crossentropy                          |
|  Optimizer: Adam                                         |
|  Metrics: accuracy                                       |
+----------------------------------------------------------+
                        |
                        v
+----------------------------------------------------------+
|                   Train the Model                        |
|  Tools: Keras (model.fit)                                |
|  Why: Trains the model using the training data with      |
|       specified batch size, epochs, and augmentation     |
+----------------------------------------------------------+
|  Batch Size: 32                                          |
|  Number of Epochs: 20                                    |
|  Save training history for later analysis                |
+----------------------------------------------------------+
                        |
                        v
+----------------------------------------------------------+
|            Evaluate the Model on Test Data               |
|  Tools: Keras (model.evaluate)                           |
|  Why: Evaluates model performance on the test dataset    |
|       to calculate accuracy, precision, recall, etc.     |
+----------------------------------------------------------+
                        |
                        v
+----------------------------------------------------------+
|                Save the Trained Model                    |
|  Tools: Keras (model.save)                               |
|  Why: Save the model for future use or deployment        |
+----------------------------------------------------------+
|  Use model.save('model_name.h5')                         |
+----------------------------------------------------------+
                        |
                        v
+----------------------------------------------------------+
|               Plot Training Results                      |
|  Tools: Matplotlib, Seaborn                              |
|  Why: Visualize training and validation loss and accuracy|
|       to monitor the learning process                    |
+----------------------------------------------------------+
|  Plot loss and accuracy for training vs validation sets  |
+----------------------------------------------------------+
                        |
                        v
+----------------------------------------------------------+
|            Make Predictions on New Data                  |
|  Tools: Keras (model.predict), NumPy                     |
|  Why: Make predictions on new/unseen data to validate    |
|       the model’s ability to generalize                  |
+----------------------------------------------------------+
                        |
                        v
+----------------------------------------------------------+
|      Generate Confusion Matrix and Classification Report |
|  Tools: Scikit-learn (confusion_matrix, classification_report) |
|  Why: To better understand the model's performance with  |
|       precision, recall, F1-score, and class-wise errors |
+----------------------------------------------------------+
                        |
                        v
+----------------------------------------------------------+
|                       End                                |
+----------------------------------------------------------+


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

