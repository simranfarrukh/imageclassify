# Image Classification using TensorFlow and Keras
Build and train a Neural Network model to classify items of clothing. 
Built by following the tutorial on the TensorFlow website for Keras Basic Classification. 

TensorFlow is a library, developed by Google, for the Deep Learning developer Community, for making deep learning applications accessible and usable to the public. Open Sourced and available on GitHub.

Keras is also a high-level (easy to use) API, built by Google AI Developer/Researcher, Francois Chollet. Written in Python and capable of running on top of backend engines like TensorFlow, CNTK, or Theano.

"tf.keras" is the Tensorflow specific implementation of the Keras API specification. It adds the framework and support for many Tensorflow specific features like perfect support for tf.data.Dataset as input objects, eager execution, etc.

## System  Requirements
Use the pip install command to install the following imports:
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```

## Usage (description of actions performed)
```
1. Import Fashion MNIST dataset
2. Explore the data
3. Preprocess the data
4. Build the model (using tf.keras.Sequential)
5. Train the model
 -  feed training data to the model: train_images & train_labels arrays
 -  model learns to associate images and labels
 -  verify that the predictions match labels from test_labels array
6. Use the trained model to make predictions
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
