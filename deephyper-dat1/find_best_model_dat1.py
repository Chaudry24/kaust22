import numpy as np
import nest_asyncio
import collections
import tensorflow as tf
from deephyper.nas import KSearchSpace
from deephyper.nas.node import ConstantNode, VariableNode
from deephyper.nas.operation import operation, Zero, Connect, AddByProjecting, Identity
from deephyper.problem import NaProblem
from deephyper.nas.preprocessing import minmaxstdscaler
from deephyper.nas.preprocessing import stdscaler
import multiprocessing
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import LoggerCallback
from deephyper.nas.run import run_base_trainer
from deephyper.search.nas import Random
import pandas as pd
import json


nest_asyncio.apply()


def load_data():
    # load train/val data
    with open("../data-competition-1a/train_x1.npy", mode="rb") as file:
        train_x = np.load(file)
    with open("../data-competition-1a/train_y1.npy", mode="rb") as file:
        train_y = np.load(file)
    with open("../data-competition-1a/val_x1.npy", mode="rb") as file:
        val_x = np.load(file)
    with open("../data-competition-1a/val_y1.npy", mode="rb") as file:
        val_y = np.load(file)

    # make a tuple for train/val data
    return (train_x, train_y), (val_x, val_y)


Activation = operation(tf.keras.layers.Activation)
Dense = operation(tf.keras.layers.Dense)
Dropout = operation(tf.keras.layers.Dropout)
Add = operation(tf.keras.layers.Add)
# Flatten = operation(tf.keras.layers.Flatten)

ACTIVATIONS = [
    # tf.keras.activations.elu,
    # tf.keras.activations.gelu,
    # tf.keras.activations.hard_sigmoid,
    tf.keras.activations.linear,
    tf.keras.activations.relu,
    # tf.keras.activations.selu,
    # tf.keras.activations.sigmoid,
    # tf.keras.activations.softplus,
    # tf.keras.activations.softsign,
    # tf.keras.activations.swish,
    # tf.keras.activations.tanh,
]


class ResNetMLPSpace(KSearchSpace):

    def __init__(self, input_shape, output_shape, seed=None, num_layers=5, mode="regression"):
        super().__init__(input_shape, output_shape, seed=seed)

        self.num_layers = num_layers
        assert mode in ["regression", "classification"]
        self.mode = mode

    def build(self):

        source = self.input_nodes[0]
        output_dim = self.output_shape[0]

        out_sub_graph = self.build_sub_graph(source, self.num_layers)

        if self.mode == "regression":
            output = ConstantNode(op=Dense(output_dim))
            self.connect(out_sub_graph, output)
        else:
            output = ConstantNode(
                op=Dense(output_dim, activation="softmax")
            )  # One-hot encoding
            self.connect(out_sub_graph, output)

        return self

    def build_sub_graph(self, input_, num_layers=3):
        source = prev_input = input_

        # look over skip connections within a range of the 3 previous nodes
        anchor_points = collections.deque([source], maxlen=3)

        for _ in range(self.num_layers):
            dense = VariableNode()
            self.add_dense_to_(dense)
            self.connect(prev_input, dense)
            x = dense

            dropout = VariableNode()
            self.add_dropout_to_(dropout)
            self.connect(x, dropout)
            x = dropout

            add = ConstantNode()
            add.set_op(AddByProjecting(self, [x], activation="relu"))

            for anchor in anchor_points:
                skipco = VariableNode()
                skipco.add_op(Zero())
                skipco.add_op(Connect(self, anchor))
                self.connect(skipco, add)

            prev_input = add

            # ! for next iter
            anchor_points.append(prev_input)

        return prev_input

    def add_dense_to_(self, node):
        node.add_op(Identity())  # we do not want to create a layer in this case
        for units in range(16, 16 * 16 + 1, 16):
            for activation in ACTIVATIONS:
                node.add_op(Dense(units=units, activation=activation))

    def add_dropout_to_(self, node):
        a, b = 1e-3, 0.1
        node.add_op(Identity())
        dropout_range = np.exp(np.random.uniform(np.log(a), np.log(b), 10))  # ! NAS
        for rate in dropout_range:
            node.add_op(Dropout(rate))


# Create a Neural Architecture problem
problem = NaProblem()

# Link the load-data function
problem.load_data(load_data)

# The function passed to preprocessing has to return
# a scikit-learn like preprocessor.
# problem.preprocessing(minmaxstdscaler)
problem.preprocessing(stdscaler)

# Link the defined search space
problem.search_space(ResNetMLPSpace)

# Fixed hyperparameters for all trained models
problem.hyperparameters(
    batch_size=32,
    learning_rate=1e-6,
    optimizer="adam",
    num_epochs=1000,
    callbacks=dict(
        EarlyStopping=dict(
            monitor="val_loss", mode="min", verbose=0, patience=5
        )
    ),
)

# Define the optimized loss (it can also be a function)
problem.loss("mse")

# Define metrics to compute for each training and validation epoch
# problem.metrics(["r2"])
problem.metrics(["rmse"])

# Define the maximised objective
# problem.objective("val_r2__last")
# problem.objective("val_loss")
problem.objective("val_acc")

# print the problem
problem

num_cpus = multiprocessing.cpu_count()

print(f"{num_cpus} CPU{'s' if num_cpus > 1 else ''} are available on this system.")

evaluator = Evaluator.create(run_base_trainer,
                             method="ray",
                             method_kwargs={
                                 # Start a new Ray server
                                 "address": None,
                                 # Defines the number of available CPUs
                                 "num_cpus": num_cpus,
                                 # Defines the number of CPUs for each task
                                 "num_cpus_per_task": 1,
                                 "callbacks": [LoggerCallback()]
                             })

print("Number of workers: ", evaluator.num_workers)


search = Random(problem, evaluator)

# find 500 best models
results = search.search(500)

# define space and shapes variable
shapes = dict(input_shape=(2,), output_shape=(1,))
space = ResNetMLPSpace(**shapes).build()


results = pd.read_csv("results.csv")
best_config = results.iloc[results.objective.argmax()][:-2].to_dict()
arch_seq = json.loads(best_config["arch_seq"])
model = space.sample(arch_seq)

# save the best model in best_model directory
model.save("./best_model")
