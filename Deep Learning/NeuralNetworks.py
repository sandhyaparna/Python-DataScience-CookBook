# https://github.com/sandhyaparna/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb
# https://github.com/aamini/introtodeeplearning_labs/blob/master/lab2/Part1_mnist.ipynb

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
  
  
import tensorflow as tf
from tf.keras.layers import *

import tensorflow as tf
print(tf.__version__)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

mnist = tf.keras.datasets.mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images=training_images/255.0
test_images=test_images/255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])


inputs = Inputs(n)
hidden = Dense(d1)(inputs)
outputs = Dense(2)(hidden)
model = Model(inputs, outputs)



