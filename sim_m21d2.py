import numpy as np
import tensorflow as tf
import models
from sklearn.preprocessing import PolynomialFeatures

DEGREE = 10


def load_data():
    # load train/val data
    with open("./data-competition-1a/train_x2.npy", mode="rb") as file:
        train_x = np.load(file)
        poly = PolynomialFeatures(degree=DEGREE, include_bias=False)
        train_x = poly.fit_transform(train_x)
    with open("./data-competition-1a/train_y2.npy", mode="rb") as file:
        train_y = np.load(file)
    with open("./data-competition-1a/test_2.npy", mode="rb") as file:
        test = np.load(file)

    # make a tuple for train/val data
    return (train_x, train_y), test


# get the data
(train_x, train_y), test = load_data()

# get the best model
model = models.model21(lr=1e-6, input_shape=train_x.shape[1])


# convert to tensors
train_x = tf.convert_to_tensor(train_x)
train_y = tf.convert_to_tensor(train_y)
test = tf.convert_to_tensor(test)

# fit the model for 1000 epochs
model.fit(x=train_x, y=train_y, epochs=1000)

# get the predictions for test set
preds = model.predict(x=test)

# save the model predictions as .npy
with open("./results/preds_m21d2.npy", mode="wb") as file:
    np.save(file, preds)

# open validation data to get validation error
with open("./data-competition-1a/val_x2.npy", mode="rb") as file:
    val_x = tf.convert_to_tensor(np.load(file))
    poly = PolynomialFeatures(degree=DEGREE, include_bias=False)
    val_x = poly.fit_transform(val_x)
with open("./data-competition-1a/val_y2.npy", mode="rb") as file:
    val_y = tf.convert_to_tensor(np.load(file))

# get loss on validation set
val_results = np.array(model.evaluate(x=val_x, y=val_y))

# save the above result
with open("./results/val_m21d2.npy", mode="wb") as file:
    np.save(file, val_results)

