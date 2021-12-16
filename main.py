# TensorFlow and tf.keras
# Concept: build & train a Neural Network model to classify images of clothing
# @author Simran
# @version 1.0

# tf.keras is a high-level API used to build and train models
import tensorflow as tf

# other libraries
import numpy as np  # working with arrays
import matplotlib.pyplot as plt

print(tf.__version__)  # for verification

# --> Import Fashion MNIST dataset <--
# access the Fashion MNIST directy from TensorFlow
fashion_mnist = tf.keras.datasets.fashion_mnist
# loading the dataset returns 4 NumPy arrays
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# class names are not included with the dataset
# so, store them under class_names for later use
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# --> Explore the data <--
train_images.shape  # 60,000 images in training set (28 x 28 pixels)
len(train_labels)  # 60,000 labels in training set
train_labels  # each label is an integer (0 — 9)
test_images.shape  # 10,000 images in test set (28 x 28 pixels)
len(test_labels)  # 10,000 image labels in test set

# --> Preprocess the data <--
# inspect first image in training set to see that pixel values
# fall in range 0 to 255
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# scale values to range of 0 to 1 before feeding them to neural network model
train_images = train_images / 255.0
test_images = test_images / 255.0

# display first 25 images from training set to verify the data is in correct format
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])  # display class name below each image
plt.show()

# --> Build the model <--
# to build the neural network we need to configure layers of model then compile it

# -> Set up the layers
model = tf.keras.Sequential([
    # transforms format of images from two dimensional array (28 x 28 pixels)
    # to one dimensional array (28 * 28 = 784 pixels)
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # each node contains a score that indicates
    # whether current image belongs to one of 10 classes
    tf.keras.layers.Dense(128, activation='relu'),  # 128 nodes
    tf.keras.layers.Dense(10)  # returns logits array with length of 10
])

# -> Compile the model
model.compile(optimizer='adam',  # model updated based on data it sees & its loss function
              # measures how accurate model during training; minimise this function
              # to steer the model in the right direction
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              # used to monitor training and testing steps;
              # uses accuracy for fraction of images correctly classified
              metrics=['accuracy'])

# -> Train the model
# 1. feed training data to the model: train_images & train_labels arrays
# 2. model learns to associate images and labels
# 3. ask model to make predictions about test set: test_images array
# 4. verify the predictions match labels from test_labels array

# > 1. Feed the model
model.fit(train_images, train_labels, epochs=10)  # loss & accuracy metrics displayed as model trains

# > 2. Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# > 3. Make predictions
# use the trained model to make prediction about some images
# softmax layer converts the logits to probabilities which are easier to interpret
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# model has predicted the label for each image in the testing set
predictions[0]  # look at the first prediction

# prediction = array of 10 numbers
# represents the model's confidence that the image corresponds
# to each of the 10 different clothing options
# you can see which label has highest confidence value
np.argmax(predictions[0])

# model is most confident that this image is ankle boot (class_names[9])
# examining test label shows that the classification is correct
test_labels[0]


# graph to look at full set of 10 class predictions
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#4f4f4f')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# > 4. Verify predictions
# use trained model to make predictions about some images
# look at 0th image, predictions, and prediction array
# correct prediction labels are blue and incorrect are red
# number gives % for predicted label
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

# look at 12th image, predictions, and prediction array
i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

# plot several images with predictions (model can be wrong even when very confident)
# Plot the first X test images, their prediction labels, and the true label
# Color correct predicitons in blue and incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_images, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# --> Use the trained model <--
# use trained to make prediction about single image
# grab image from test dataset
img = test_images[1]

print(img.shape)

# add image to a batch where it's the only member
img = (np.expand_dims(img, 0))

print(img.shape)

# predict the correct label for image
predictions_single = probability_model.predict(img)

print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

# tf.keras.Model.predict returns list of list (one list for each image in batch)
# grab the prediction for only our image in batch
np.argmax(predictions_single[0])
