import tensorflow as tf


def rmse_loss(y_pred, y_true):
    return tf.sqrt(tf.keras.losses.MeanSquaredError()(y_pred, y_true))


def model1(dense_units=1024, lr=1e-3, n_layers=5):
    """This function returns a compiled
    sequential neural network"""

    # initialize empty list to store model layers
    l = []

    # append input layer
    l.append(tf.keras.layers.Input(shape=(2)))

    # add layers
    for i in range(n_layers):
        l.append(tf.keras.layers.Dense(units=dense_units,
                                       activation="relu"))

    # append the output layer
    l.append(tf.keras.layers.Dense(units=1))

    # initialize model
    model = tf.keras.Sequential(l)

    # compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def model2(dense_units=1024, lr=1e-3):
    """This function returns a compiled NN with skip connections"""

    # input layer
    inputs = tf.keras.layers.Input(shape=(2))

    # dense layers
    d1 = tf.keras.layers.Dense(units=dense_units, activation="relu")(inputs)
    d2 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d1)

    # concat output of d1 with d2 to pass into d3
    c12 = tf.keras.layers.Concatenate()([d1, d2])
    d3 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c12)

    # concatenate output of d2 with d3 to pass into d4
    c23 = tf.keras.layers.Concatenate()([d2, d3])
    d4 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c23)

    # concatenate d3 with d4 to pass into d5
    c34 = tf.keras.layers.Concatenate()([d3, d4])
    d5 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c34)

    # pass d5 into output
    output = tf.keras.layers.Dense(units=1)(d5)

    # connect the inputs and outputs
    model = tf.keras.Model(inputs, output)

    # compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def model3(dense_units=1024, lr=1e-3):
    """This function returns a compiled NN with skip connections"""

    # input layer
    inputs = tf.keras.layers.Input(shape=(2))

    # dense layers
    d1 = tf.keras.layers.Dense(units=dense_units, activation="relu")(inputs)
    d2 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d1)
    d3 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d2)
    d4 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d3)

    # concatenate inputs with d4 to pass into d5
    c34 = tf.keras.layers.Concatenate()([inputs, d4])
    d5 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c34)

    # pass d5 into output
    output = tf.keras.layers.Dense(units=1)(d5)

    # connect the inputs and outputs
    model = tf.keras.Model(inputs, output)

    # compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def model4(dense_units=1024, lr=1e-3):
    """This function returns a compiled NN with skip connections"""

    # input layer
    inputs = tf.keras.layers.Input(shape=(2))

    # dense layers
    d1 = tf.keras.layers.Dense(units=dense_units, activation="relu")(inputs)
    d2 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d1)
    d3 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d2)
    d4 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d3)

    # concatenate d1 with d4 to pass into d5
    c34 = tf.keras.layers.Concatenate()([d1, d4])
    d5 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c34)

    # pass d5 into output
    output = tf.keras.layers.Dense(units=1)(d5)

    # connect the inputs and outputs
    model = tf.keras.Model(inputs, output)

    # compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def model5(dense_units=1024, lr=1e-3):
    """This function returns a compiled NN with skip connections"""

    # input layer
    inputs = tf.keras.layers.Input(shape=(2))

    # dense layers
    d1 = tf.keras.layers.Dense(units=dense_units, activation="relu")(inputs)
    d2 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d1)
    d3 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d2)
    d4 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d3)

    # concatenate d2 with d4 to pass into d5
    c34 = tf.keras.layers.Concatenate()([d2, d4])
    d5 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c34)

    # pass d5 into output
    output = tf.keras.layers.Dense(units=1)(d5)

    # connect the inputs and outputs
    model = tf.keras.Model(inputs, output)

    # compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def model6(dense_units=1024, lr=1e-3):
    """This function returns a compiled NN with skip connections"""

    # input layer
    inputs = tf.keras.layers.Input(shape=(2))

    # dense layers
    d1 = tf.keras.layers.Dense(units=dense_units, activation="relu")(inputs)
    d2 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d1)
    d3 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d2)
    d4 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d3)

    # concatenate d3 with d4 to pass into d5
    c34 = tf.keras.layers.Concatenate()([d3, d4])
    d5 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c34)

    # pass d5 into output
    output = tf.keras.layers.Dense(units=1)(d5)

    # connect the inputs and outputs
    model = tf.keras.Model(inputs, output)

    # compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def model7(dense_units=1024, lr=1e-3):
    """This function returns a compiled NN with skip connections"""

    # input layer
    inputs = tf.keras.layers.Input(shape=(2))

    # dense layers
    d1 = tf.keras.layers.Dense(units=dense_units, activation="relu")(inputs)
    d2 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d1)

    # concat output of d1 with d2 to pass into d3
    c12 = tf.keras.layers.Concatenate()([d1, d2])
    d3 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c12)

    # concatenate output of d2 with d3 to pass into d4
    c23 = tf.keras.layers.Concatenate()([d2, d3])
    d4 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c23)

    # concatenate d3 with d4 to pass into d5
    c34 = tf.keras.layers.Concatenate()([d3, d4])
    d5 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c34)

    # concatenate inputs with d5 and pass into output
    cind5 = tf.keras.layers.Concatenate()([inputs, d5])
    d6 = tf.keras.layers.Dense(units=dense_units, activation="relu")(cind5)

    # pass d5 into output
    output = tf.keras.layers.Dense(units=1)(d6)

    # connect the inputs and outputs
    model = tf.keras.Model(inputs, output)

    # compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def model8(dense_units=1024, lr=1e-3):
    """This function returns a compiled NN with skip connections"""

    # input layer
    inputs = tf.keras.layers.Input(shape=(2))

    # dense layers
    d1 = tf.keras.layers.Dense(units=dense_units, activation="relu")(inputs)
    d2 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d1)

    # concat output of d1 with d2 to pass into d3
    c12 = tf.keras.layers.Add()([d1, d2])
    d3 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c12)

    # concatenate output of d2 with d3 to pass into d4
    c23 = tf.keras.layers.Add()([d2, d3])
    d4 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c23)

    # concatenate d3 with d4 to pass into d5
    c34 = tf.keras.layers.Add()([d3, d4])
    d5 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c34)

    # pass d5 into output
    output = tf.keras.layers.Dense(units=1)(d5)

    # connect the inputs and outputs
    model = tf.keras.Model(inputs, output)

    # compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def model9(dense_units=1024, lr=1e-3):
    """This function returns a compiled NN with skip connections"""

    # input layer
    inputs = tf.keras.layers.Input(shape=(2))

    # dense layers
    d1 = tf.keras.layers.Dense(units=dense_units, activation="relu")(inputs)
    d2 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d1)
    d3 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d2)
    d4 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d3)

    # concatenate inputs with d4 to pass into d5
    c34 = tf.keras.layers.Concatenate()([inputs, d4])
    d5 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c34)

    # pass d5 into output
    output = tf.keras.layers.Dense(units=1)(d5)

    # connect the inputs and outputs
    model = tf.keras.Model(inputs, output)

    # compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def model10(dense_units=1024, lr=1e-3):
    """This function returns a compiled NN with skip connections"""

    # input layer
    inputs = tf.keras.layers.Input(shape=(2))

    # dense layers
    d1 = tf.keras.layers.Dense(units=dense_units, activation="relu")(inputs)
    d2 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d1)
    d3 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d2)
    d4 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d3)

    # concatenate d1 with d4 to pass into d5
    c34 = tf.keras.layers.Add()([d1, d4])
    d5 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c34)

    # pass d5 into output
    output = tf.keras.layers.Dense(units=1)(d5)

    # connect the inputs and outputs
    model = tf.keras.Model(inputs, output)

    # compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def model11(dense_units=1024, lr=1e-3):
    """This function returns a compiled NN with skip connections"""

    # input layer
    inputs = tf.keras.layers.Input(shape=(2))

    # dense layers
    d1 = tf.keras.layers.Dense(units=dense_units, activation="relu")(inputs)
    d2 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d1)
    d3 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d2)
    d4 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d3)

    # concatenate d2 with d4 to pass into d5
    c34 = tf.keras.layers.Add()([d2, d4])
    d5 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c34)

    # pass d5 into output
    output = tf.keras.layers.Dense(units=1)(d5)

    # connect the inputs and outputs
    model = tf.keras.Model(inputs, output)

    # compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def model12(dense_units=1024, lr=1e-3):
    """This function returns a compiled NN with skip connections"""

    # input layer
    inputs = tf.keras.layers.Input(shape=(2))

    # dense layers
    d1 = tf.keras.layers.Dense(units=dense_units, activation="relu")(inputs)
    d2 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d1)
    d3 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d2)
    d4 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d3)

    # concatenate d3 with d4 to pass into d5
    c34 = tf.keras.layers.Add()([d3, d4])
    d5 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c34)

    # pass d5 into output
    output = tf.keras.layers.Dense(units=1)(d5)

    # connect the inputs and outputs
    model = tf.keras.Model(inputs, output)

    # compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def model13(dense_units=1024, lr=1e-3):
    """This function returns a compiled NN with skip connections"""

    # input layer
    inputs = tf.keras.layers.Input(shape=(2))

    # dense layers
    d1 = tf.keras.layers.Dense(units=dense_units, activation="relu")(inputs)
    d2 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d1)

    # concat output of d1 with d2 to pass into d3
    c12 = tf.keras.layers.Add()([d1, d2])
    d3 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c12)

    # concatenate output of d2 with d3 to pass into d4
    c23 = tf.keras.layers.Add()([d2, d3])
    d4 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c23)

    # concatenate d3 with d4 to pass into d5
    c34 = tf.keras.layers.Add()([d3, d4])
    d5 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c34)

    # concatenate inputs with d5 and pass into output
    cind5 = tf.keras.layers.Concatenate()([inputs, d5])
    d6 = tf.keras.layers.Dense(units=dense_units, activation="relu")(cind5)

    # pass d5 into output
    output = tf.keras.layers.Dense(units=1)(d6)

    # connect the inputs and outputs
    model = tf.keras.Model(inputs, output)

    # compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def model14(dense_units=1024, lr=1e-3):
    """This function returns a compiled NN with skip connections"""

    # input layer
    inputs = tf.keras.layers.Input(shape=(2))

    # dense layers
    d1 = tf.keras.layers.Dense(units=dense_units, activation="relu")(inputs)
    d2 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d1)
    d3 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d2)
    d4 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d3)

    # concatenate d3 with d4 to pass into d5
    c34 = tf.keras.layers.Concatenate()([d3, d4])
    d5 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c34)
    d5 = tf.keras.layers.Dropout(0.10)(d5)

    # pass d5 into output
    output = tf.keras.layers.Dense(units=1)(d5)

    # connect the inputs and outputs
    model = tf.keras.Model(inputs, output)

    # compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def model15(dense_units=1024, lr=1e-3):
    """This function returns a compiled NN with skip connections"""

    # input layer
    inputs = tf.keras.layers.Input(shape=(2))

    # dense layers
    d1 = tf.keras.layers.Dense(units=dense_units, activation="relu")(inputs)
    d2 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d1)

    # concat output of d1 with d2 to pass into d3
    c12 = tf.keras.layers.Concatenate()([d1, d2])
    d3 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c12)

    # concatenate output of d2 with d3 to pass into d4
    c23 = tf.keras.layers.Concatenate()([d2, d3])
    d4 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c23)

    # concatenate d3 with d4 to pass into d5
    c34 = tf.keras.layers.Concatenate()([d3, d4])
    d5 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c34)

    # concatenate inputs with d5 and pass into output
    cind5 = tf.keras.layers.Concatenate()([inputs, d5])
    d6 = tf.keras.layers.Dense(units=dense_units, activation="relu")(cind5)
    d6 = tf.keras.layers.Dropout(0.10)(d6)

    # pass d5 into output
    output = tf.keras.layers.Dense(units=1)(d6)

    # connect the inputs and outputs
    model = tf.keras.Model(inputs, output)

    # compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def model16(dense_units=1024, lr=1e-3):
    """This function returns a compiled NN with skip connections"""

    # input layer
    inputs = tf.keras.layers.Input(shape=(2))

    # dense layers
    d1 = tf.keras.layers.Dense(units=dense_units, activation="relu")(inputs)
    d2 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d1)
    d3 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d2)
    d4 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d3)
    d4 = tf.keras.layers.Dropout(0.05)(d4)

    # concatenate d3 with d4 to pass into d5
    c34 = tf.keras.layers.Concatenate()([d3, d4])
    d5 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c34)
    d5 = tf.keras.layers.Dropout(0.05)(d5)

    # pass d5 into output
    output = tf.keras.layers.Dense(units=1)(d5)

    # connect the inputs and outputs
    model = tf.keras.Model(inputs, output)

    # compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def model17(dense_units=1024, lr=1e-3):
    """This function returns a compiled NN with skip connections"""

    # input layer
    inputs = tf.keras.layers.Input(shape=(2))

    # dense layers
    d1 = tf.keras.layers.Dense(units=dense_units, activation="relu")(inputs)
    d2 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d1)

    # concat output of d1 with d2 to pass into d3
    c12 = tf.keras.layers.Concatenate()([d1, d2])
    d3 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c12)

    # concatenate output of d2 with d3 to pass into d4
    c23 = tf.keras.layers.Concatenate()([d2, d3])
    d4 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c23)

    # concatenate d3 with d4 to pass into d5
    c34 = tf.keras.layers.Concatenate()([d3, d4])
    d5 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c34)
    d5 = tf.keras.layers.Dropout(0.05)(d5)

    # concatenate inputs with d5 and pass into output
    cind5 = tf.keras.layers.Concatenate()([inputs, d5])
    d6 = tf.keras.layers.Dense(units=dense_units, activation="relu")(cind5)
    d6 = tf.keras.layers.Dropout(0.05)(d6)

    # pass d5 into output
    output = tf.keras.layers.Dense(units=1)(d6)

    # connect the inputs and outputs
    model = tf.keras.Model(inputs, output)

    # compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def model18(dense_units=1024, lr=1e-3):
    """This function returns a compiled NN with skip connections"""

    # input layer
    inputs = tf.keras.layers.Input(shape=(2))

    # dense layers
    d1 = tf.keras.layers.Dense(units=dense_units, activation="relu")(inputs)
    d1 = tf.keras.layers.Dropout(0.05)(d1)
    d2 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d1)

    # concat output of d1 with d2 to pass into d3
    c12 = tf.keras.layers.Concatenate()([d1, d2])
    d3 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c12)

    # concatenate output of d2 with d3 to pass into d4
    c23 = tf.keras.layers.Concatenate()([d2, d3])
    d4 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c23)

    # concatenate d3 with d4 to pass into d5
    c34 = tf.keras.layers.Concatenate()([d3, d4])
    d5 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c34)
    d5 = tf.keras.layers.Dropout(0.05)(d5)

    # concatenate inputs with d5 and pass into output
    cind5 = tf.keras.layers.Concatenate()([inputs, d5])
    d6 = tf.keras.layers.Dense(units=dense_units, activation="relu")(cind5)
    d6 = tf.keras.layers.Dropout(0.05)(d6)

    # pass d5 into output
    output = tf.keras.layers.Dense(units=1)(d6)

    # connect the inputs and outputs
    model = tf.keras.Model(inputs, output)

    # compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def model19(dense_units=1024, lr=1e-3):
    """This function returns a compiled NN with skip connections"""

    # input layer
    inputs = tf.keras.layers.Input(shape=(2))

    # dense layers
    d1 = tf.keras.layers.Dense(units=128, activation="relu")(inputs)
    d1 = tf.keras.layers.Dropout(0.05)(d1)
    d2 = tf.keras.layers.Dense(units=256, activation="relu")(d1)

    # concat output of d1 with d2 to pass into d3
    c12 = tf.keras.layers.Concatenate()([d1, d2])
    d3 = tf.keras.layers.Dense(units=512, activation="relu")(c12)

    # concatenate output of d2 with d3 to pass into d4
    c23 = tf.keras.layers.Concatenate()([d2, d3])
    d4 = tf.keras.layers.Dense(units=1024, activation="relu")(c23)

    # concatenate d3 with d4 to pass into d5
    c34 = tf.keras.layers.Concatenate()([d3, d4])
    d5 = tf.keras.layers.Dense(units=512, activation="relu")(c34)
    d5 = tf.keras.layers.Dropout(0.05)(d5)

    # concatenate inputs with d5 and pass into output
    cind5 = tf.keras.layers.Concatenate()([inputs, d5])
    d6 = tf.keras.layers.Dense(units=256, activation="relu")(cind5)
    d6 = tf.keras.layers.Dropout(0.05)(d6)

    # pass d5 into output
    output = tf.keras.layers.Dense(units=1)(d6)

    # connect the inputs and outputs
    model = tf.keras.Model(inputs, output)

    # compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def model20(dense_units=1024, lr=1e-3):
    """This function returns a compiled NN with skip connections"""

    # input layer
    inputs = tf.keras.layers.Input(shape=(2))

    # dense layers
    d1 = tf.keras.layers.Dense(units=dense_units, activation="relu")(inputs)
    d2 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d1)

    # concat output of d1 with d2 to pass into d3
    c12 = tf.keras.layers.Concatenate()([d1, d2])
    d3 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c12)

    # concatenate output of d2 with d3 to pass into d4
    c23 = tf.keras.layers.Concatenate()([d2, d3])
    d4 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c23)

    # concatenate d3 with d4 to pass into d5
    c34 = tf.keras.layers.Concatenate()([d3, d4])
    d5 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c34)
    d5 = tf.keras.layers.Dropout(0.05)(d5)

    # concatenate inputs with d5 and pass into output
    cind5 = tf.keras.layers.Concatenate()([inputs, d5])
    d6 = tf.keras.layers.Dense(units=dense_units, activation="relu")(cind5)
    d6 = tf.keras.layers.Dropout(0.05)(d6)
    d7 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d6)
    d7 = tf.keras.layers.Dropout(0.05)(d7)

    # pass into output
    output = tf.keras.layers.Dense(units=1)(d7)

    # connect the inputs and outputs
    model = tf.keras.Model(inputs, output)

    # compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def model21(dense_units=1024, lr=1e-3):
    """This function returns a compiled NN with skip connections"""

    # input layer
    inputs = tf.keras.layers.Input(shape=(2))

    # dense layers
    d1 = tf.keras.layers.Dense(units=dense_units, activation="relu")(inputs)
    d2 = tf.keras.layers.Dense(units=dense_units, activation="relu")(d1)

    # concat output of d1 with d2 to pass into d3
    c12 = tf.keras.layers.Concatenate()([d1, d2])
    d3 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c12)

    # concatenate output of d2 with d3 to pass into d4
    c23 = tf.keras.layers.Concatenate()([d2, d3])
    d4 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c23)

    # concatenate d3 with d4 to pass into d5
    c34 = tf.keras.layers.Concatenate()([d3, d4])
    d5 = tf.keras.layers.Dense(units=dense_units, activation="relu")(c34)
    d5_drp = tf.keras.layers.Dropout(0.05)(d5)

    # concatenate inputs with d5 and pass into d6
    cind5 = tf.keras.layers.Concatenate()([inputs, d5])
    d6 = tf.keras.layers.Dense(units=dense_units, activation="relu")(cind5)
    d6_drp = tf.keras.layers.Dropout(0.05)(d6)

    cd5_drpd6 = tf.keras.layers.Concatenate()([d5_drp, d6])
    d7 = tf.keras.layers.Dense(units=dense_units, activation="relu")(cd5_drpd6)

    cd6_drpd7 = tf.keras.layers.Concatenate()([d6_drp, d7])
    d8 = tf.keras.layers.Dense(units=dense_units, activation="relu")(cd6_drpd7)
    d8 = tf.keras.layers.Dropout(0.05)(d8)

    # pass into output
    output = tf.keras.layers.Dense(units=1)(d8)

    # connect the inputs and outputs
    model = tf.keras.Model(inputs, output)

    # compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


