import numpy as np
import tensorflow as tf
import models


def load_data():
    # load train/val data
    with open("./data-competition-1a/train_x2.npy", mode="rb") as file:
        train_x = np.load(file)
    with open("./data-competition-1a/train_y2.npy", mode="rb") as file:
        train_y = np.load(file)
    with open("./data-competition-1a/test_2.npy", mode="rb") as file:
        test = np.load(file)

    # make a tuple for train/val data
    return (train_x, train_y), test


# get the best model
model = models.model20(lr=1e-8)

# get the data
(train_x, train_y), test = load_data()

# convert to tensors
train_x = tf.convert_to_tensor(train_x)
train_y = tf.convert_to_tensor(train_y)
test = tf.convert_to_tensor(test)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                     min_delta=0,
                                     patience=100,
                                     verbose=0,
                                     mode='min',
                                     baseline=None,
                                     restore_best_weights=True)
]

# open validation data to get validation error
with open("./data-competition-1a/val_x2.npy", mode="rb") as file:
    val_x = tf.convert_to_tensor(np.load(file))
with open("./data-competition-1a/val_y2.npy", mode="rb") as file:
    val_y = tf.convert_to_tensor(np.load(file))

# fit the model for 1000 epochs
model.fit(x=train_x, y=train_y, epochs=1000,
          validation_split=0.1,
          callbacks=callbacks)

# get the predictions for test set
preds = model.predict(x=test)

# save the model predictions as .npy
with open("./results/preds_m20d2.npy", mode="wb") as file:
    np.save(file, preds)

# get loss on validation set
val_results = np.array(model.evaluate(x=val_x, y=val_y))

# save the above result
with open("./results/val_m20d2.npy", mode="wb") as file:
    np.save(file, val_results)

