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
  
### Neural Networks
# keras is a Tensorflow API
# Dense - layers of connection
# Units - Number of Neurons
# Sequential - Successive layers are defined in Sequence
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














