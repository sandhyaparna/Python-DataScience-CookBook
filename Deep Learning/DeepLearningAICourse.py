### Google Colab Notebooks
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%201%20-%20House%20Prices/Exercise_1_House_Prices_Answer.ipynb#scrollTo=PUNO2E6SeURH
# 


# https://github.com/lmoroney/dlaicourse
* Training Neural Networks 
  * model.compile to set optimizer and loss function
  * model.fit on train data
  * model.evaluate on test data gives loss and acc
  * model.predict gives the probability of targets 0,1,2 etc
* Image DataSet 
  * How to load data that is already present in tf.keras datasets API as training and test images and their labels
  * How to standardize data from images for Neural networks
  * model.predict of a particular observation gives the probability of that obs being classifies as 0,1,2,3 etc (depends on number of target labels)
  * As number of neurons in tf.keras.layers.Dense layer increases - accuracy of model inc along with time to train
  * Number of Neurons in the last layer should always match the number of classes u are classifying for
  * First layer in your network should be the same shape as your data
  * callbacks is used to stop training on more epochs if desired value of loss or accuracy is obtained
* CNN on Image
  * Convolution is added on top of NN in model
  * Filters generated in tf.keras.layers.Conv2D are not random. They start with a set of known good filters in a similar 
    way to the pattern fitting that you saw earlier, and the ones that work from that set are learned over time.
  * tf.keras.layers.MaxPooling2D(2,2) implies max value out of the 4 values of 2*2 matrix survives
 
### Neural Networks
# keras is a Tensorflow API
# Dense - layers of connection
# Units - Number of Neurons
# Sequential - Successive layers are defined in Sequence
import tensorflow as tf
import numpy as np
from tensorflow import keras
# NN with 1 layer and that layer has 1 neuron and the input shape to it is just 1 value
model = keras.sequential([keras.layers.Dense(units=1, input_shape=[1])])

# 3 layers
# First layer corresponds to input - shape to be expected for the data to be in. Flatten is used to take image to convert it into a simple array
# Each image in MNIST data is represented as 28*28 array of (rows and columns)
# Last layer corresponds to diff Target classes
# Hidden Layer - 128 neurons - 
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),  #tf.keras.layers.Flatten(), - flattening is imp in images
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), #more hidden layers
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# for image data - Normalize the input values

# Loss functions - Measure how good the current guess is
# Optimizer - First NN starts with a guess and then optimizer is used to improve upon it. It generates a new and improved guess
# (sgd - stochastic gradient descent)
model.compile(optimizer='sgd', #'adam'
              loss='mean_squared_error', #'sparse_categorical_crossentropy'
              metrics=['accuracy']) #

# xs - Input data
# ys - Target var
# epochs - Number of training loops 
model.fit(xs, ys, epochs=500)
model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

model.predict(test_images)
model.predict([y])

# Stops training based on callbacks
# callbacks is based on loss
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True
callbacks = myCallback()
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

# callbacks is based on accuracy
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

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

# Visualizing layers - Journey of an image through Convolution
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










