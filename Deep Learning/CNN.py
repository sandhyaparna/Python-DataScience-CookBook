# https://github.com/aamini/introtodeeplearning_labs/blob/master/lab2/Part1_mnist_solution.ipynb
# https://lodev.org/cgtutor/filtering.html

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

training_images=training_images.reshape(60000, 28, 28, 1) #reshaping is done? probably do accomodate colour
test_images = test_images.reshape(10000, 28, 28, 1)

training_images=training_images / 255.0
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

### Image Generator API
In many practical applications, All images will not be of same size/pixels. 
ImageGenerator is passed into a directory that contains each unique Target set as subdirectories
Within Horses_Humans - Training and Validation folders are present - And each of the two will have Horses, Humans sub directories
Rescale is used to normalize data

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Note that because we are facing a two-class classification problem, i.e. a binary classification problem,
# we will end our network with a sigmoid activation
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)), # 3 bytes for colour, 1 for blue, 1 for green and another for red
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid') #1 neuron for Sigmoid handles binary label
])
model.summary()

# In this case, using the RMSprop optimization algorithm is preferable to stochastic gradient descent (SGD), because 
# RMSprop automates learning-rate tuning for us. (Other optimizers, such as Adam and Adagrad, also automatically adapt
# the learning rate during training, and would work equally well here.)
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])


from tensorflow.keras.preprocessing.image import ImageDataGenerator
# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

#Image pre-processing
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
                 # This is the source directory for training images - Names of Subdirectories will be the labels of the images
        'C:/Users/sandh/Google Drive/Data Science/Python Learning/DataSets/Images/Horses_Humans/Train_Horse_Human/',  
        target_size=(300, 300),  # All images will be resized to 150x150. Input data of NN should all be of same size
        batch_size=128,
           # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
        'C:/Users/sandh/Google Drive/Data Science/Python Learning/DataSets/Images/Horses_Humans/Validation_Horse_Human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')
# instead of 300*300 if the images are resized to 150*150 - the training is quicker

from tensorflow.keras.optimizers import RMSprop

# fit the model
history = model.fit_generator( # model.fit generator is used instead of datasets
      train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1)

# to get validation loss as well
history = model.fit_generator(
      train_generator,
      steps_per_epoch=8,  
      epochs=15,
      validation_data=validation_generator,
      validation_steps=8,
      verbose=2)
   
  
### kFold - in tensorflow
# https://stackoverflow.com/questions/38164798/does-tensorflow-have-cross-validation-implemented-for-its-users?rq=1
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Parameters
learning_rate = 0.01
batch_size = 500

# TF graph
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
pred = tf.nn.softmax(tf.matmul(x, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()

mnist = input_data.read_data_sets("data/mnist-tf", one_hot=True)
train_x_all = mnist.train.images
train_y_all = mnist.train.labels
test_x = mnist.test.images
test_y = mnist.test.labels

def run_train(session, train_x, train_y):
  print "\nStart training"
  session.run(init)
  for epoch in range(10):
    total_batch = int(train_x.shape[0] / batch_size)
    for i in range(total_batch):
      batch_x = train_x[i*batch_size:(i+1)*batch_size]
      batch_y = train_y[i*batch_size:(i+1)*batch_size]
      _, c = session.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
      if i % 50 == 0:
        print "Epoch #%d step=%d cost=%f" % (epoch, i, c)

def cross_validate(session, split_size=5):
  results = []
  kf = KFold(n_splits=split_size)
  for train_idx, val_idx in kf.split(train_x_all, train_y_all):
    train_x = train_x_all[train_idx]
    train_y = train_y_all[train_idx]
    val_x = train_x_all[val_idx]
    val_y = train_y_all[val_idx]
    run_train(session, train_x, train_y)
    results.append(session.run(accuracy, feed_dict={x: val_x, y: val_y}))
  return results

with tf.Session() as session:
  result = cross_validate(session)
  print "Cross-validation result: %s" % result
  print "Test accuracy: %f" % session.run(accuracy, feed_dict={x: test_x, y: test_y})
  
  
  
  
  

