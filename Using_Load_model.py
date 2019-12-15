#Handwritten numbers classification
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#Explore data
print(y_train[12])
print(np.shape(x_train))
print(np.shape(x_test))
#we have 60000 imae for the training and 10000 for testing
#print(x_train[12])

# We should normalize data
x_train = x_train/255
x_test = x_test/255
 #reshape the data
#Plot data
plt.imshow(x_train[12],cmap =plt.cm.binary)
#plt.show()
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
y_train = y_train.reshape(60000,1)
y_test = y_test.reshape(10000,1)
print(np.shape(x_train))
#Load model

model = load_model("CNN_MNIST.h5")

test_loss,test_acc = model.evaluate(x_test,y_test)
print("\ntest accuracy:",test_acc)