#Handwritten numbers classification
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
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
plt.show()
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
y_train = y_train.reshape(60000,1)
y_test = y_test.reshape(10000,1)
print(np.shape(x_train))
#Create a model
model = keras.Sequential([
	#keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Conv2D(4,(3,3),(1,1),padding = "same",input_shape=(28,28,1)),
	keras.layers.MaxPooling2D(pool_size = (2,2),padding = "valid"),
	#keras.layers.Conv2D(32,(3,3),(1,1),padding = "same"),
	#keras.layers.MaxPooling2D(pool_size = (2,2),padding = "valid"),
	keras.layers.Flatten(),
	keras.layers.Dense(128,activation = "relu"),
	keras.layers.Dense(10,activation = "softmax")])

model.compile(optimizer = "adam",
	loss = "sparse_categorical_crossentropy",
	metrics  = ['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x_train, y_train, epochs=3, validation_split=0.2, callbacks=[early_stopping])
test_loss,test_acc = model.evaluate(x_test,y_test)
print("\ntest accuracy:",test_acc)

model.save(r"C:\Users\HP\Desktop\Opencv training\CNN_MNIST_4filters.h5")