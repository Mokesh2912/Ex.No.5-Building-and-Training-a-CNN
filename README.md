# EX.NO.5 – BUILDING AND TRAINING A CNN

## TITLE

**Building and Training a Convolutional Neural Network (CNN) using TensorFlow for Image Classification**

---

## AIM

To build and train a **Convolutional Neural Network (CNN)** using the **TensorFlow/Keras framework** and perform image classification using the **CIFAR-10 dataset**.

---

## OBJECTIVES

1. To understand the fundamentals of Convolutional Neural Networks.
2. To load the CIFAR-10 image dataset using TensorFlow/Keras.
3. To preprocess and normalize image data.
4. To perform one-hot encoding of class labels.
5. To visualize sample images from the dataset.
6. To construct a CNN using convolutional and pooling layers.
7. To use dropout layers to reduce overfitting.
8. To compile the CNN using an appropriate optimizer and loss function.
9. To train the CNN model using training data.
10. To visualize training and validation accuracy and loss.
11. To classify unseen test images using the trained CNN.

---

# THEORY

## 1. Convolutional Neural Network

A **Convolutional Neural Network (CNN)** is a deep learning model designed primarily for processing image data.

Unlike traditional machine-learning methods, CNNs can automatically learn useful image features directly from pixel values.

A typical CNN consists of:

* Input layer
* Convolutional layers
* Activation functions
* Pooling layers
* Dropout layers
* Flatten layer
* Fully connected/Dense layers
* Output layer

---

## 2. Convolutional Layer

A convolutional layer applies filters or kernels to an image.

The filters move across the image and detect important spatial patterns such as:

* Edges
* Shapes
* Textures
* Object parts

In this experiment, `Conv2D` layers are used.

Example:

```python
layers.Conv2D(32, (3,3), activation='relu')
```

This creates 32 filters of size 3 × 3.

---

## 3. ReLU Activation Function

The **Rectified Linear Unit (ReLU)** activation function is used in the convolutional and dense layers.

It is defined as:

$$
ReLU(x)=max(0,x)
$$

ReLU introduces non-linearity into the neural network and helps the network learn complex patterns.

---

## 4. Max Pooling

Max pooling reduces the spatial dimensions of feature maps.

In this experiment:

```python
layers.MaxPooling2D((2,2))
```

is used.

It selects the maximum value from each 2 × 2 region.

The advantages are:

* Reduces computational complexity
* Reduces feature-map dimensions
* Retains important features
* Helps improve model efficiency

---

## 5. Dropout

Dropout is a regularization technique used to reduce overfitting.

During training, randomly selected neurons are temporarily ignored.

In this experiment, dropout values of **0.25** and **0.5** are used.

---

## 6. Flatten Layer

After convolution and pooling, the feature maps are converted into a one-dimensional vector using:

```python
layers.Flatten()
```

This allows the extracted image features to be passed to the dense layers.

---

## 7. Dense Layer

Dense layers perform the final classification using the features extracted by the convolutional layers.

The experiment uses:

```python
layers.Dense(512, activation='relu')
```

followed by the final output layer.

---

## 8. Softmax Output Layer

The final layer contains **10 neurons**, one for each CIFAR-10 class.

```python
layers.Dense(10, activation='softmax')
```

Softmax converts the outputs into probability values.

The class having the highest probability is selected as the predicted class.

---

# DATASET

## CIFAR-10 Dataset

CIFAR-10 is an image dataset containing **60,000 color images** belonging to ten different classes.

The dataset consists of:

| Dataset      | Number of Images |
| ------------ | ---------------: |
| Training Set |           50,000 |
| Test Set     |           10,000 |
| Total        |           60,000 |

Each image has:

* Width = 32 pixels
* Height = 32 pixels
* Channels = 3 RGB

Therefore, the input shape is:

```text
(32, 32, 3)
```

---

## CIFAR-10 Classes

| Class Number | Class      |
| -----------: | ---------- |
|            0 | Airplane   |
|            1 | Automobile |
|            2 | Bird       |
|            3 | Cat        |
|            4 | Deer       |
|            5 | Dog        |
|            6 | Frog       |
|            7 | Horse      |
|            8 | Ship       |
|            9 | Truck      |

---

# SOFTWARE REQUIREMENTS

### Hardware Requirements

* Computer/Laptop
* Minimum 4 GB RAM
* GPU is optional

### Software Requirements

* Python 3.x
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Jupyter Notebook / Google Colab

---

# PROCEDURE

## STEP 1: IMPORT LIBRARIES

Import the required libraries.

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
```

### Purpose

* TensorFlow/Keras → Build and train the CNN
* NumPy → Numerical operations
* Matplotlib → Visualization

---

# STEP 2: LOAD THE CIFAR-10 DATASET

The CIFAR-10 dataset is loaded using Keras.

```python
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
```

Display the shapes:

```python
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
```

### Expected Output

```text
Training data shape: (50000, 32, 32, 3)
Testing data shape: (10000, 32, 32, 3)
```

---

# STEP 3: PREPROCESS THE DATA

## 3.1 Normalize the Images

The original image pixels have values between 0 and 255.

They are converted into floating-point values and scaled to the range 0 to 1.

```python
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
```

### Purpose

Normalization:

* Improves numerical stability
* Helps faster convergence
* Makes training more efficient

---

## 3.2 One-Hot Encode the Labels

The ten class labels are converted into one-hot encoded vectors.

```python
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

For example, a class is represented using a vector of length 10.

This format is suitable for categorical cross-entropy classification.

---

# STEP 4: VISUALIZE SAMPLE IMAGES

Define the CIFAR-10 class names.

```python
class_names = [
    'Airplane',
    'Automobile',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck'
]
```

Display 16 sample images.

```python
plt.figure(figsize=(10,10))

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    plt.xlabel(class_names[np.argmax(y_train[i])])

plt.show()
```

### Observation

A 4 × 4 grid of CIFAR-10 images is displayed.

Each image is labeled with its corresponding class.

---

# STEP 5: BUILD THE CNN MODEL

The CNN is constructed using the Keras Sequential model.

```python
model = models.Sequential()
```

---

## First Convolution Block

```python
model.add(
    layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        padding='same',
        input_shape=(32,32,3)
    )
)

model.add(
    layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        padding='same'
    )
)

model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))
```

### Function

The first convolution block extracts basic features from the images.

---

## Second Convolution Block

```python
model.add(
    layers.Conv2D(
        64,
        (3,3),
        activation='relu',
        padding='same'
    )
)

model.add(
    layers.Conv2D(
        64,
        (3,3),
        activation='relu',
        padding='same'
    )
)

model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))
```

### Function

This block learns more complex features from the images.

---

## Third Convolution Block

```python
model.add(
    layers.Conv2D(
        128,
        (3,3),
        activation='relu',
        padding='same'
    )
)

model.add(
    layers.Conv2D(
        128,
        (3,3),
        activation='relu',
        padding='same'
    )
)

model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))
```

### Function

This block extracts higher-level and more complex visual features.

---

## Fully Connected Classification Layers

```python
model.add(layers.Flatten())

model.add(
    layers.Dense(
        512,
        activation='relu'
    )
)

model.add(layers.Dropout(0.5))

model.add(
    layers.Dense(
        10,
        activation='softmax'
    )
)
```

The Flatten layer converts the feature maps into a one-dimensional vector.

The Dense layer performs classification, and the final Softmax layer produces the ten class probabilities.

---

# CNN ARCHITECTURE

The complete architecture is:

```text
Input Image
32 × 32 × 3
      ↓
Conv2D – 32 filters
      ↓
Conv2D – 32 filters
      ↓
MaxPooling
      ↓
Dropout
      ↓
Conv2D – 64 filters
      ↓
Conv2D – 64 filters
      ↓
MaxPooling
      ↓
Dropout
      ↓
Conv2D – 128 filters
      ↓
Conv2D – 128 filters
      ↓
MaxPooling
      ↓
Dropout
      ↓
Flatten
      ↓
Dense – 512 neurons
      ↓
Dropout
      ↓
Dense – 10 neurons
      ↓
Softmax Output
```

---

# STEP 6: COMPILE THE MODEL

The model is compiled using the Adam optimizer and categorical cross-entropy loss.

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

Display the model architecture:

```python
model.summary()
```

---

## Optimizer

### Adam

Adam is used for optimizing the neural-network weights during training.

It provides efficient gradient-based optimization and is commonly used for deep-learning models.

---

## Loss Function

### Categorical Cross-Entropy

Categorical cross-entropy is suitable for multiclass classification when the target labels are one-hot encoded.

---

## Metric

### Accuracy

Accuracy measures the proportion of correctly classified samples.

$$
Accuracy =
\frac{\text{Number of Correct Predictions}}
{\text{Total Number of Predictions}}
$$

---

# STEP 7: TRAIN THE MODEL

The CNN is trained for 30 epochs using a batch size of 64.

A validation split of 20% is used.

```python
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.2
)
```

### Training Parameters

| Parameter        | Value                     |
| ---------------- | ------------------------- |
| Epochs           | 30                        |
| Batch Size       | 64                        |
| Validation Split | 20%                       |
| Optimizer        | Adam                      |
| Loss Function    | Categorical Cross-Entropy |
| Activation       | ReLU and Softmax          |

---

# STEP 8: PLOT TRAINING AND VALIDATION ACCURACY

```python
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.plot(
    history.history['accuracy'],
    label='Train Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
```

---

# STEP 9: PLOT TRAINING AND VALIDATION LOSS

```python
plt.subplot(1,2,2)

plt.plot(
    history.history['loss'],
    label='Train Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
```

---

# OBSERVATION – ACCURACY GRAPH

The training accuracy generally increases as the number of epochs increases, showing that the CNN is learning useful patterns from the training images.

The validation accuracy also improves during training. If the validation accuracy begins to level off while training accuracy continues increasing, this may indicate the beginning of overfitting.

The accuracy graph is therefore useful for analyzing the learning behavior of the model.

---

# OBSERVATION – LOSS GRAPH

The training loss generally decreases during training as the model becomes better at classifying the training images.

The validation loss generally decreases during the initial stages of training and may fluctuate later.

The loss graph helps determine whether the model is continuing to improve or whether training has started to stabilize.

---

# STEP 10: PREDICT TEST IMAGES

A function is created to predict the class of a test image.

```python
def plot_predictions(index):

    img = X_test[index]

    true_label = class_names[
        np.argmax(y_test[index])
    ]

    pred_probs = model.predict(
        np.expand_dims(img, axis=0),
        verbose=0
    )

    pred_label = class_names[
        np.argmax(pred_probs)
    ]

    plt.imshow(img)

    plt.title(
        f"True: {true_label} | Pred: {pred_label}"
    )

    plt.axis('off')

    plt.show()
```

Predict the first five test images:

```python
for i in range(5):
    plot_predictions(i)
```

---

# PREDICTION OUTPUT

For each test image, the program displays:

```text
True: <Actual Class> | Pred: <Predicted Class>
```

For example, the output format will be similar to:

```text
True: Airplane | Pred: Airplane
```

The exact predicted classes depend on the trained model.

---

# OPTIONAL TEST SET EVALUATION

The model can also be evaluated directly on the complete test dataset.

```python
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
```

This gives the final loss and accuracy of the trained model on the unseen CIFAR-10 test set.

---

# COMPLETE PROGRAM

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------
# STEP 1: LOAD CIFAR-10 DATASET
# -------------------------------------------------

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# -------------------------------------------------
# STEP 2: NORMALIZE IMAGE DATA
# -------------------------------------------------

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# -------------------------------------------------
# STEP 3: ONE-HOT ENCODE LABELS
# -------------------------------------------------

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# -------------------------------------------------
# CLASS NAMES
# -------------------------------------------------

class_names = [
    'Airplane',
    'Automobile',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck'
]

# -------------------------------------------------
# STEP 4: VISUALIZE SAMPLE IMAGES
# -------------------------------------------------

plt.figure(figsize=(10,10))

for i in range(16):

    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    plt.imshow(X_train[i])

    plt.xlabel(
        class_names[np.argmax(y_train[i])]
    )

plt.show()

# -------------------------------------------------
# STEP 5: BUILD CNN MODEL
# -------------------------------------------------

model = models.Sequential()

# First convolution block
model.add(
    layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        padding='same',
        input_shape=(32,32,3)
    )
)

model.add(
    layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        padding='same'
    )
)

model.add(
    layers.MaxPooling2D((2,2))
)

model.add(
    layers.Dropout(0.25)
)

# Second convolution block
model.add(
    layers.Conv2D(
        64,
        (3,3),
        activation='relu',
        padding='same'
    )
)

model.add(
    layers.Conv2D(
        64,
        (3,3),
        activation='relu',
        padding='same'
    )
)

model.add(
    layers.MaxPooling2D((2,2))
)

model.add(
    layers.Dropout(0.25)
)

# Third convolution block
model.add(
    layers.Conv2D(
        128,
        (3,3),
        activation='relu',
        padding='same'
    )
)

model.add(
    layers.Conv2D(
        128,
        (3,3),
        activation='relu',
        padding='same'
    )
)

model.add(
    layers.MaxPooling2D((2,2))
)

model.add(
    layers.Dropout(0.25)
)

# Fully connected layers
model.add(
    layers.Flatten()
)

model.add(
    layers.Dense(
        512,
        activation='relu'
    )
)

model.add(
    layers.Dropout(0.5)
)

# Output layer
model.add(
    layers.Dense(
        10,
        activation='softmax'
    )
)

# -------------------------------------------------
# STEP 6: COMPILE MODEL
# -------------------------------------------------

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------------------------
# STEP 7: TRAIN MODEL
# -------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.2
)

# -------------------------------------------------
# STEP 8: PLOT ACCURACY
# -------------------------------------------------

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.plot(
    history.history['accuracy'],
    label='Train Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# -------------------------------------------------
# STEP 9: PLOT LOSS
# -------------------------------------------------

plt.subplot(1,2,2)

plt.plot(
    history.history['loss'],
    label='Train Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# -------------------------------------------------
# STEP 10: TEST SET EVALUATION
# -------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# -------------------------------------------------
# STEP 11: PREDICT TEST IMAGES
# -------------------------------------------------

def plot_predictions(index):

    img = X_test[index]

    true_label = class_names[
        np.argmax(y_test[index])
    ]

    pred_probs = model.predict(
        np.expand_dims(img, axis=0),
        verbose=0
    )

    pred_label = class_names[
        np.argmax(pred_probs)
    ]

    plt.imshow(img)

    plt.title(
        f"True: {true_label} | Pred: {pred_label}"
    )

    plt.axis('off')

    plt.show()


# Predict first five test images
for i in range(5):
    plot_predictions(i)
```

---

# ALGORITHM

1. Start the program.
2. Import TensorFlow, Keras, NumPy and Matplotlib.
3. Load the CIFAR-10 dataset.
4. Separate the training and testing images.
5. Normalize the image pixel values to the range 0 to 1.
6. Convert the class labels into one-hot encoded vectors.
7. Define the ten CIFAR-10 class names.
8. Display sample images from the training dataset.
9. Create a Sequential CNN model.
10. Add convolutional layers with ReLU activation.
11. Add max-pooling layers to reduce feature-map dimensions.
12. Add dropout layers to reduce overfitting.
13. Flatten the extracted feature maps.
14. Add a dense layer with 512 neurons.
15. Add a final dense layer with 10 neurons and Softmax activation.
16. Compile the model using the Adam optimizer.
17. Use categorical cross-entropy as the loss function.
18. Train the model for 30 epochs with a batch size of 64.
19. Use 20% of the training data for validation.
20. Plot training and validation accuracy.
21. Plot training and validation loss.
22. Evaluate the model on the test dataset.
23. Predict the classes of test images.
24. Display the actual and predicted labels.
25. Stop the program.

---

# FLOW OF THE CNN

```text
CIFAR-10 Image
      ↓
Data Normalization
      ↓
One-Hot Encoding
      ↓
Convolution Layer
      ↓
ReLU Activation
      ↓
Convolution Layer
      ↓
Max Pooling
      ↓
Dropout
      ↓
Convolution Layer
      ↓
Max Pooling
      ↓
Dropout
      ↓
Convolution Layer
      ↓
Max Pooling
      ↓
Dropout
      ↓
Flatten
      ↓
Dense Layer
      ↓
Dropout
      ↓
Softmax
      ↓
10-Class Prediction
```

---

# OBSERVATION TABLE

| Parameter         | Observation               |
| ----------------- | ------------------------- |
| Dataset           | CIFAR-10                  |
| Training Images   | 50,000                    |
| Testing Images    | 10,000                    |
| Image Size        | 32 × 32                   |
| Channels          | 3 RGB                     |
| Number of Classes | 10                        |
| Optimizer         | Adam                      |
| Loss Function     | Categorical Cross-Entropy |
| Epochs            | 30                        |
| Batch Size        | 64                        |
| Validation Split  | 20%                       |
| Output Activation | Softmax                   |

---

# PERFORMANCE ANALYSIS

The CNN can be analyzed using the following:

### Training Accuracy

Shows how well the CNN classifies images used during training.

### Validation Accuracy

Shows how well the model performs on the validation portion of the training dataset.

### Training Loss

Indicates the error made by the model on training data.

### Validation Loss

Indicates the error on validation data.

### Test Accuracy

Measures classification performance on the unseen test dataset.

---

# OVERFITTING ANALYSIS

Overfitting occurs when the model performs very well on training data but does not generalize equally well to unseen data.

The accuracy and loss graphs can be used to identify this behavior.

The experiment uses **Dropout** layers to help reduce overfitting.

A validation split of 20% is also used to monitor generalization during training.

---

# EXPECTED OUTPUTS

The following outputs are obtained:

### Output 1 – Dataset Shape

```text
Training data shape: (50000, 32, 32, 3)
Testing data shape: (10000, 32, 32, 3)
```

### Output 2 – Sample Image Grid

A 4 × 4 grid containing 16 CIFAR-10 sample images with their class names.

### Output 3 – Model Summary

The CNN architecture and the number of parameters in each layer are displayed.

### Output 4 – Training Progress

The model is trained for 30 epochs.

### Output 5 – Accuracy Graph

A graph containing:

* Training Accuracy
* Validation Accuracy

### Output 6 – Loss Graph

A graph containing:

* Training Loss
* Validation Loss

### Output 7 – Test Evaluation

The program displays:

```text
Test Loss: ...
Test Accuracy: ...
```

The exact numerical values depend on the actual training run.

### Output 8 – Prediction

Five test images are displayed with:

```text
True: <class> | Pred: <class>
```

---

# RESULT

Thus, a **Convolutional Neural Network (CNN)** was successfully built and trained using **TensorFlow/Keras** and the **CIFAR-10 dataset**. The CNN learned image features through convolutional and pooling layers and performed multiclass image classification for ten different object categories.

---

# CONCLUSION

Thus, the CNN was successfully implemented and trained to recognize objects belonging to ten distinct categories using the CIFAR-10 dataset. The experiment demonstrated the use of convolutional layers, ReLU activation, max-pooling, dropout, dense layers and Softmax classification. Training and validation accuracy and loss were visualized, and the trained model was used to classify unseen test images.

