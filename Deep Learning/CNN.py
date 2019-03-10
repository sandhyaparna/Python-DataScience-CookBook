# https://github.com/aamini/introtodeeplearning_labs/blob/master/lab2/Part1_mnist_solution.ipynb
# 

* CNN on Image
  * Convolution is added on top of NN in model
  * Filters generated in tf.keras.layers.Conv2D are not random. They start with a set of known good filters in a similar 
    way to the pattern fitting that you saw earlier, and the ones that work from that set are learned over time.
  * tf.keras.layers.MaxPooling2D(2,2) implies max value out of the 4 values of 2*2 matrix survives

### CNN
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)), #Generate 64 filters, each of 3*3 matrix, 
                                                       # ReLU activation, 1 in input shape is where we are mentioning colour depth
  tf.keras.layers.MaxPooling2D(2, 2),  # max value out of the 4 values of 2*2 matrix survives
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

import tensorflow as tf
print(tf.__version__)
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(training_images, training_labels, epochs=5)
test_loss = model.evaluate(test_images, test_labels)

### Visualizing layers - Journey of an image through Convolution
import matplotlib.pyplot as plt
f, axarr = plt.subplots(3,4) #3 corresponds to the number of images of same fashion item passed
                             #4 is because there are 2 convolutional layers and 2 max pooling layers
FIRST_IMAGE=0 #Shoe is present as 1st obs in test image, 
SECOND_IMAGE=7 #Shoe is also present as 8th obs
THIRD_IMAGE=26 #Shoe is also present as 26th obs
CONVOLUTION_NUMBER = 1 #filter 1 or 2 or 3 or 4 etc
from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):
  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)



