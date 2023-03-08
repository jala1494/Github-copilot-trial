
#Solving MNIST using tensorflow keras and a CNN
#import all the modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

#load the dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

#normalize the data
train_images = train_images/255.0
test_images = test_images/255.0
#Reshape the data
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)



#Do a one hot encoding of the labels
train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

#Create a CNN using the sequential API
model = tf.keras.models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

#Compile the model
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
     metrics=['accuracy']
    )

#Train the model and keep the history. use a validation split of 0.2
history = model.fit(
    train_images, train_labels,
    validation_split=0.2,
    epochs=5)
#Render the history. Use loss and validation loss.
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

#Render accuracy and validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

#Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)






