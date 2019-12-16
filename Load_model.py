#Handwritten numbers classification
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


model = load_model(r"C:\Users\HP\Desktop\Opencv training\CNN_MNIST_4filters.h5")

# Let's visualize the learned filters of our model
#Summary of the model
model.summary()

#
for layer in model.layers:
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# get filter weights
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)

f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)


# plot first few filters
n_filters, ix = 4, 1
for i in range(n_filters):
	# get the filter
	f = filters[:, :, :, i]
	# plot each channel separately
	#for j in range(3):
		# specify subplot and turn of axis
	ax = plt.subplot(n_filters, 2, ix)
	ax.set_xticks([])
	ax.set_yticks([])
		# plot filter channel in grayscale
	plt.imshow(f[:, :, 0], cmap='gray')
	ix += 1
# show the figure
plt.show()