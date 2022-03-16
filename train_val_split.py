import pandas as pd
import numpy as np
import sklearn.model_selection
import os

os.chdir("../")

"""THIS SCRIPT IS USED TO GET A RANDOM TRAIN VALIDATION SPLIT OF THE DATA"""

# load datasets
train_data1 = pd.read_csv("./data-competition-1a/1a_1_train.csv")
train_data2 = pd.read_csv("./data-competition-1a/1a_2_train.csv")

# get the train/val splits
train1, val1 = sklearn.model_selection.train_test_split(train_data1, test_size=0.08, shuffle=True)
train2, val2 = sklearn.model_selection.train_test_split(train_data2, test_size=0.08, shuffle=True)

# save the new datasets as csv
train1.to_csv('./data-competition-1a/train1.csv', index=False)
val1.to_csv('./data-competition-1a/val1.csv', index=False)

train2.to_csv('./data-competition-1a/train2.csv', index=False)
val2.to_csv('./data-competition-1a/val2.csv', index=False)

# convert the new datasets to numpy arrays
train_x1 = train1.to_numpy()[:, [1, 2]]
train_y1 = train1.to_numpy()[:, -1]
val_x1 = val1.to_numpy()[:, [1, 2]]
val_y1 = val1.to_numpy()[:, -1]

train_x2 = train2.to_numpy()[:, [1, 2]]
train_y2 = train2.to_numpy()[:, -1]
val_x2 = val2.to_numpy()[:, [1, 2]]
val_y2 = val2.to_numpy()[:, -1]

# save the numpy arrays as .npy
with open("./data-competition-1a/train_x1.npy", mode="wb") as file:
    np.save(file, train_x1)
with open("./data-competition-1a/train_y1.npy", mode="wb") as file:
    np.save(file, train_y1)
with open("./data-competition-1a/val_x1.npy", mode="wb") as file:
    np.save(file, val_x1)
with open("./data-competition-1a/val_y1.npy", mode="wb") as file:
    np.save(file, val_y1)


with open("./data-competition-1a/train_x2.npy", mode="wb") as file:
    np.save(file, train_x2)
with open("./data-competition-1a/train_y2.npy", mode="wb") as file:
    np.save(file, train_y2)
with open("./data-competition-1a/val_x2.npy", mode="wb") as file:
    np.save(file, val_x2)
with open("./data-competition-1a/val_y2.npy", mode="wb") as file:
    np.save(file, val_y2)


# load test datasets
test_1 = pd.read_csv("./data-competition-1a/1a_1_test.csv")
test_2 = pd.read_csv("./data-competition-1a/1a_2_test.csv")

# convert test datasets to numpy
test_1 = test_1.to_numpy()[:, [1, 2]]
test_2 = test_2.to_numpy()[:, [1, 2]]

# save the test datasets as .npy
with open("./data-competition-1a/test_1.npy", mode="wb") as file:
    np.save(file, test_1)
with open("./data-competition-1a/test_2.npy", mode="wb") as file:
    np.save(file, test_2)


