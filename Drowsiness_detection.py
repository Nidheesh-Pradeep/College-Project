#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, BatchNormalization, MaxPooling2D,Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow import keras
import numpy as np 
import pandas as pd 
import os
from glob import glob
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from PIL import Image


# In[2]:


# Update the file paths to the appropriate directories
closed_eyes_dir = 'C:/DISSERTATION/Dataset/train/Closed_Eyes'
open_eyes_dir = 'C:/DISSERTATION/Dataset/train/Open_Eyes'

# Function to load images from a directory
def load_images_from_dir(directory, label):
    X = []
    Y = []
    for filename in tqdm(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        temp = np.array(Image.open(filepath).resize((80, 80)))
        X.append(temp)
        Y.append(label)
    return X, Y

# Load images from the 'Closed_Eyes' directory with label 1
X_closed, Y_closed = load_images_from_dir(closed_eyes_dir, 1)

# Load images from the 'Open_Eyes' directory with label 0
X_open, Y_open = load_images_from_dir(open_eyes_dir, 0)

# Combine the data from both classes
X = X_closed + X_open
Y = Y_closed + Y_open

 


# In[3]:


# Normalize X to have values between 0 and 1 using Min-Max scaling
X = (np.array(X) - np.min(X)) / (np.max(X) - np.min(X))

# Further scale X by dividing it by 255 to ensure values are in the range [0, 1]
X = X / 255.0

# Normalize Y (assuming Y is a separate array) to have values between 0 and 1 using Min-Max scaling
Y = (np.array(Y) - np.min(Y)) / (np.max(Y) - np.min(Y))

# At this point, X and Y have been normalized to the [0, 1] range

# Check the shape of array X
X.shape


# In[4]:


# Expand the dimensions of array X along the last axis (-1) to make it compatible for certain operations
X = np.expand_dims(X, -1)

# Check the new shape of array X
X.shape


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

# Data augmentation parameters
dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10,
    horizontal_flip=True,  # Enable horizontal flipping
    vertical_flip=False,    # Disable vertical flipping
    fill_mode='nearest'     # Fill mode for newly created pixels
)

# Fit the data generator on your dataset
dataGen.fit(X)
newbatches = dataGen.flow(X, Y,batch_size=4000)  # REQUESTING DATA GENRATOR TO GENERATE IMAGES  BATCH SIZE = NO. OF IMAGES CREAtED EACH TIME ITS CALLED
X1, Y1 = next(newbatches)

# Number of augmented images to generate per original image
num_augmented_images = 4  # You can adjust this number as needed


# Perform data augmentation and visualize the results
for i, (x, y) in enumerate(zip(X, Y)):
    x = np.expand_dims(x, axis=0)  # Add a batch dimension
    y = np.expand_dims(y, axis=0)  # Add a batch dimension

    # Generate augmented images
    augmented_images = []
    for _ in range(num_augmented_images):
        for x_aug, y_aug in dataGen.flow(x, y, batch_size=1):
            augmented_images.append((x_aug[0], y_aug[0]))
            break  # Generate only one augmented image per loop

    # Visualize the original and augmented images
    plt.figure(figsize=(12, 4))
    plt.subplot(1, num_augmented_images + 1, 1)
    plt.imshow(x[0])
    plt.title("Original")
    plt.axis('off')

    for j, (x_aug, y_aug) in enumerate(augmented_images):
        plt.subplot(1, num_augmented_images + 1, j + 2)
        plt.imshow(x_aug)
        plt.title(f"Augmented {j + 1}")
        plt.axis('off')
        
        
    plt.tight_layout()
    plt.show()

    if i >= 4:  # Limit the number of original images to visualize
        break


# In[7]:


# Split the data into training and testing sets
# x_train and y_train will be used for training the model
# x_test and y_test will be used for evaluating the model's performance
x_train, x_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.33, random_state=42)
# Further split the training data into training and validation sets
# x_train and y_train will be used for training the model
# x_validation and y_validation will be used for tuning hyperparameters and validation
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.33, random_state=42)


# In[8]:


# Print the shapes of the different datasets
print('\033[1m' + 'Data Shapes')

# Print the shape of the training dataset
print("Train data", end="")
print(x_train.shape, y_train.shape)

# Print the shape of the validation dataset
print("Validation data", end="")
print(x_validation.shape, y_validation.shape)

# Print the shape of the testing dataset
print("Test data", end="")
print(x_test.shape, y_test.shape)


# In[9]:


import tensorflow as tf
from tensorflow import keras

# Instantiate the model
model = keras.Sequential()

# Adding first two  convolutional layers
model.add(Conv2D(
    filters=32, # number of filters
    kernel_size=(5, 5), # height/width of filter
    strides=1, # Step size when sliding the filter over the input
    activation='relu', # activation function
    input_shape=(80, 80, 1)  # Input shape for the first layer
))
model.add(Conv2D(
    filters=32,
    kernel_size=(5, 5),
    strides=1,
    activation='relu',
    input_shape=(80, 80, 1),
    use_bias=False # Whether to use bias terms in the convolutional layer
))
model.add(BatchNormalization()) # Apply batch normalization to normalize activations
model.add(MaxPooling2D(strides=2)) # Apply max pooling to reduce spatial dimensions
model.add(Dropout(0.3)) # Apply dropout to prevent overfitting

# Adding the second set of convolutional layers
model.add(Conv2D(
    filters=64,
    kernel_size=(5, 5),
    strides=1,
    activation='relu',
    input_shape=(80, 80, 1)
))
model.add(Conv2D(
    filters=64,
    kernel_size=(5, 5),
    strides=1,
    activation='relu',
    input_shape=(80, 80, 1),
    use_bias=False
))
model.add(BatchNormalization())
model.add(MaxPooling2D(strides=2))
model.add(Dropout(0.3))

# Flatten the feature maps for the dense layers
model.add(Flatten())

# Adding the first dense layer with 256 nodes
model.add(Dense(units=256, activation='relu', use_bias=False))
model.add(BatchNormalization()) # Apply batch normalization

# Adding the second dense layer with 128 nodes
model.add(Dense(units=128, use_bias=False, activation='relu'))
model.add(BatchNormalization()) # Apply batch normalization


# Adding the third dense layer with 84 nodes
model.add(Dense(units=84, use_bias=False, activation='relu'))
model.add(BatchNormalization())


model.add(Dropout(0.3))


# Adding the output layer with 1 unit and sigmoid activation
model.add(Dense(units=1, activation='sigmoid'))

# compiling the model
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])


# In[10]:


#print the model summary 
print(model.summary())


# In[38]:


from tensorflow.keras.callbacks import EarlyStopping
# Define a ModelCheckpoint callback to save the best model during training
callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='bestModel80.h5', # Path to save the best model
    save_weights_only=False, # Save the entire model, including architecture
    monitor='val_loss',      # Monitor validation loss for saving
    mode='min',              # 'min' mode saves the model when validation loss decreases
    save_best_only=True,     # Save only the best model during training
    verbose=1                # Show verbose output
)

# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor (e.g., validation loss)
    patience=20,         # Number of epochs with no improvement after which training will stop
    restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored metric
)

# Train the model using the fit() function
history = model.fit(
    x_train, # Training data
    y_train, # Training labels
    validation_split=0.20, # Split data for validation (20%)
    epochs= 50, #number of aining epochs
    batch_size=32,# Batch size for training
    callbacks=[callback,early_stopping]# List of callbacks, including ModelCheckpoint
)


# In[39]:


def plot_learning_curve(history):
    # Extract loss and accuracy values
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    # Create an array of epoch numbers
    epochs = range(1, len(train_loss) + 1)
    
    
    # Plot training and validation accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
plt.show()
# Call the function to create the learning curve
plot_learning_curve(history)


# In[40]:


# Evaluate the trained model on the test data
score = model.evaluate(x_test, y_test)

# Print the test score (loss) obtained from the evaluation
print('Test Score:', score[0])

# Print the test accuracy obtained from the evaluation
print('Test Accuracy:', score[1])


# In[21]:


import matplotlib.pyplot as plt

# Evaluate the trained model on the test data
score = model.evaluate(x_test, y_test)

# Extract the test score (loss) and test accuracy
test_loss = score[0]
test_accuracy = score[1]

# Create a bar chart or line plot to visualize the test metrics
metrics = ['Loss', 'Accuracy']
values = [test_loss, test_accuracy]

# Create a bar chart
plt.bar(metrics, values, color=['red', 'blue'])
plt.ylabel('Value')
plt.title('Test Metrics')
plt.show()

# Alternatively, create a line plot
plt.plot(metrics, values, marker='o', linestyle='-')
plt.ylabel('Value')
plt.title('Test Metrics')
plt.grid(True)
plt.show()


# In[41]:


from keras.models import load_model
best_model = load_model('bestModel80.h5')
best_model.evaluate(x_test, y_test)


# In[23]:


# Import the load_model function from Keras to load a saved model
from keras.models import load_model

# Load the best model from the saved file 'bestModel.h5'
best_model = load_model('bestModel80.h5')

# Evaluate the loaded best model on the test data
evaluation_score = best_model.evaluate(x_test, y_test)

# Print the evaluation results
print('Test Loss:', evaluation_score[0])
print('Test Accuracy:', evaluation_score[1])


# In[24]:


import matplotlib.pyplot as plt

# Evaluate the trained model on the test data
score = best_model.evaluate(x_test, y_test)

# Extract the test score (loss) and test accuracy
best_modeltest_loss = score[0]
best_modeltest_accuracy = score[1]

# Create a bar chart or line plot to visualize the test metrics
best_modelmetrics = ['Loss', 'Accuracy']
best_modelvalues = [best_modeltest_loss, best_modeltest_accuracy]

# Create a bar chart
plt.bar(best_modelmetrics, best_modelvalues, color=['red', 'blue'])
plt.ylabel('Value')
plt.title('Test Metrics')
plt.show()

# Alternatively, create a line plot
plt.plot(best_modelmetrics, best_modelvalues, marker='o', linestyle='-')
plt.ylabel('Value')
plt.title('Test Metrics')
plt.grid(True)
plt.show()


# In[25]:


import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


# Evaluate the trained model on the test data
score = model.evaluate(x_test, y_test)
test_loss = score[0]
test_accuracy = score[1]

# Assuming 'model' is your trained classifier, and 'x_test' is your test data
y_scores = model.predict(x_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
auc = roc_auc_score(y_test, y_scores)

# Load the best model from the saved file 'bestModel80.h5'
best_model = load_model('bestModel80.h5')

# Evaluate the loaded best model on the test data
evaluation_score = best_model.evaluate(x_test, y_test)
best_model_loss = evaluation_score[0]
best_model_accuracy = evaluation_score[1]

# Compute ROC curve and AUC for the best model
y_scores_best = best_model.predict(x_test)
fpr_best, tpr_best, thresholds_best = roc_curve(y_test, y_scores_best)
auc_best = roc_auc_score(y_test, y_scores_best)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot(fpr_best, tpr_best, color='green', lw=2, label=f'Best Model ROC (AUC = {auc_best:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

plt.show()


# In[22]:


# Directory to save
os.chdir('C:/DISSERTATION/Codes')
         
# Save best model 
model.save('bestModel80.h5')


# In[27]:


from sklearn.metrics import f1_score

threshold_values = []  # Store threshold values
f1_scores = []         # Store F1 scores

# Iterate over different threshold values
for threshold in np.arange(0, 1.1, 0.1):  # Test thresholds from 0 to 1 in steps of 0.1
    # Apply threshold to model predictions and create binary predictions
    binary_predictions = [1 if result >= threshold else 0 for result in best_model.predict(x_validation)]

    # Calculate F1 score using the validation set
    f1 = f1_score(y_validation, binary_predictions)

    # Append threshold and F1 score to lists
    threshold_values.append(threshold)
    f1_scores.append(f1)

# Identify the optimal threshold (e.g., the one with the highest F1 score)
optimal_threshold = threshold_values[np.argmax(f1_scores)]
print("Optimal Threshold:", optimal_threshold)


# In[28]:


import random

# Shuffle the test data to make a random selection
random.shuffle(x_test)

# Select a random subset of samples (e.g., 20 samples)
random_subset = x_test[:20]

# Iterate over the random subset
for i in random_subset:
    result = best_model.predict(np.expand_dims(i, 0))
    plt.figure(figsize=(13, 7))
    plt.imshow(i)
    plt.show()

    if result < 0.2:
        print('\033[92m' + '\033[1m' + 'Eyes Open - Driver is awake and safe' + '\033[0m')
    else:
        print('\033[91m' + '\033[1m' + 'Eyes Closed - Driver is sleepy.' '\n' +
              'ALERT-Take a break and continue driving later' + '\033[0m')
        


# In[42]:


from sklearn.metrics import confusion_matrix 
import seaborn as sns

plt.figure(figsize=(27, 9))

preds = best_model.predict(x_test)
preds = (preds >= 0.2).astype(np.int32)
cm = confusion_matrix(y_test, preds)
df_cm = pd.DataFrame(cm, index=['closed', 'Open'], columns=['Closed', 'Open'])
plt.subplot(121)
plt.title("Confusion matrix\n")
sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
plt.ylabel("Predicted")
plt.xlabel("Actual")


# In[43]:


from sklearn.metrics import f1_score, accuracy_score, precision_score

# Calculate F1-score, accuracy, and precision
f1 = f1_score(y_test, preds)
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)

print("F1-score:", f1)
print("Accuracy:", accuracy)
print("Precision:", precision)


# In[44]:


accuracy = best_model.evaluate(x_test, y_test)[1]
print("Accuracy:", accuracy)


# In[ ]:




