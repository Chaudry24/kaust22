import numpy as np
import tensorflow as tf


def load_data():
    # load train/val data
    with open("../data-competition-1a/train_x2.npy", mode="rb") as file:
        train_x = np.load(file)
    with open("../data-competition-1a/train_y2.npy", mode="rb") as file:
        train_y = np.load(file)
    with open("../data-competition-1a/test_2.npy", mode="rb") as file:
        test = np.load(file)

    # make a tuple for train/val data
    return (train_x, train_y), test


# get the best model
model = tf.keras.models.load_model("./best_model")

# compile the model
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-6),
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

# get the data
(train_x, train_y), test = load_data()

# fit the model for 1000 epochs
model.fit(x=train_x, y=train_y, epochs=1000)

# get the predictions for test set
preds = model.predict(x=test)

# save the model predictions as .npy
with open("./best_model_results/preds.npy", mode="wb") as file:
    np.save(file, preds)

# open validation data to get validation error
with open("../data-competition-1a/val_x2.npy", mode="rb") as file:
    val_x = np.load(file)
with open("../data-competition-1a/val_y2.npy", mode="rb") as file:
    val_y = np.load(file)

# get loss on validation set
val_results = np.array(model.evaluate(x=val_x, y=val_y))

# save the above result
with open("./best_model_results/val_d2.npy", mode="wb") as file:
    np.save(file, preds)

