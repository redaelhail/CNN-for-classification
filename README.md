# CNN-for-classification

This is a small project of a convolutional neural network to classify handwritten digits. 

•	The main file is 'CNN_MNIST.py' wich is a python program where I built a CNN for handwritten digits classification. The dataset
was imported using Keras datasets, it is a sort of images (28x28) that contain digits. This dataset was formed of 60000 images for
training and 10000 for testing. I trained the model using adam optimizer and sparse cross entropy as a loss function in 3 epochs.

•	After the training, the accuracy of this model was above 98 percent.

•	In order to reuse the trained model, I saved it in an h5py file wich is 'CNN_MNIST_Model.py'. In order to load it, I made a little 
program 'Load_Model.py' that loads the trained model and use it to predict a new input.

•	Figures 1 and 2 are for data exploration.

If you have any advices or remarks, feel free to contact me.
