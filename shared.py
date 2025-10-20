import keras
import pandas as pd
import numpy as np
import tensorflow_federated as tff
import collections
import sklearn
import matplotlib.pyplot as plt

# MODEL PARAMETERS
hidden_layer_count  = 10
neuron_count        = 10
activation          = 'relu'
optimizer           = 'sgd'

# OVERALL PARAMETERS
client_count        = 5     # number of clients
round_count         = 10000 # Number of rounds each client runs
learning_rate       = 0.1   # Model learning rate.

scaler = sklearn.preprocessing.StandardScaler()
label_encoder = sklearn.preprocessing.LabelEncoder()

def build_model(input_shape):
    """Creates a Sequential Keras NN which will be used for classifying attacks"""
    # Define the model
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(None, input_shape[0], input_shape[1])))
    for _ in range(hidden_layer_count):
        model.add(keras.layers.Dense(neuron_count, activation=activation))
    model.add(keras.layers.Dense(6, activation='softmax'))

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def unison_shuffled_copies(a, b):
    """
    Helper function for shuffling train and test arrays in unison.
    
    Source: https://stackoverflow.com/a/4602224
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def process_data():
    train_data = pd.read_csv('train.csv')
    train_data = train_data.sample(frac=1).reset_index(drop=True) # Shuffle data.
    train_label = train_data['Label']
    train_data = train_data.drop('Label', axis=1)
    test_data = pd.read_csv('test.csv')
    test_label = test_data['Label']
    test_data = test_data.drop('Label', axis=1)

    # Convert data to all numeric types, using one-hot encoding on categorical data
    train_data = pd.get_dummies(train_data, drop_first=True)
    test_data = pd.get_dummies(test_data, drop_first=True)

    # Scale all data
    train_data = pd.DataFrame(scaler.fit_transform(train_data), columns = train_data.columns)
    test_data = pd.DataFrame(scaler.transform(test_data), columns = test_data.columns)

    # Encode labels to numbers
    train_label = label_encoder.fit_transform(train_label)
    test_label = label_encoder.transform(test_label)

    return train_data, train_label, test_data, test_label

def split_train(train_data, train_label, client_count):
    # Split training data between clients.
    split_train_data = np.array_split(train_data, client_count)
    split_train_label = np.array_split(train_label, client_count)
    return split_train_data, split_train_label

def tff_setup(model, input_spec, learning_rate):
    # Make the model federated-learning compatible for tff to use
    model = tff.learning.models.functional_model_from_keras(
        model, keras.losses.SparseCategoricalCrossentropy(), 
        input_spec,
        metrics_constructor=collections.OrderedDict(accuracy=keras.metrics.SparseCategoricalAccuracy, loss=keras.metrics.SparseCategoricalCrossentropy)
    )

    # Create the federated learning training setup
    trainer = tff.learning.algorithms.build_weighted_fed_avg(
       model, 
       client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=learning_rate)
    )

    return trainer

def federated_train(trainer, train_datasets, round_count):
    state = trainer.initialize()
    max_accuracy_result = None
    accuracy = []
    loss = []
    for i in range(round_count):
        result = trainer.next(state, train_datasets)
        state = result.state
        metrics = result.metrics
        accuracy.append(metrics['client_work']['train']['accuracy'])
        loss.append(metrics['client_work']['train']['loss'])
        if max_accuracy_result is None or \
            metrics['client_work']['train']['accuracy'] >= max_accuracy_result.metrics['client_work']['train']['accuracy']:
            max_accuracy_result = result
        # print(f"Round {i} Accuracy: {metrics['client_work']['train']['accuracy']}, Loss: {metrics['client_work']['train']['loss']}")
    return max_accuracy_result, accuracy, loss
    
def federated_evaluation(input_spec, max_accuracy_result, test_dataset, test_data, test_label, model_name):
    # Prepare an inference model based on the best performing state.
    # This will be used to predict based on the test data we set aside.
    inference_model = build_model((input_spec[0].shape))
    print(f"Best Accuracy: {max_accuracy_result.metrics['client_work']['train']['accuracy']}")
    max_accuracy_result.state.global_model_weights.assign_weights_to(inference_model)

    # Evaluate the model's performance on the test dataset.
    inference_model.evaluate(test_dataset)

    # Run some predictions on the model and generate a confusion matrix
    prediction = inference_model.predict(test_data).argmax(axis=1)
    accuracy_score = sklearn.metrics.accuracy_score(test_label, prediction)
    confusion_matrix = sklearn.metrics.confusion_matrix(test_label, prediction)
    print(model_name)
    print(f"Testing - Prediction Accuracy: {accuracy_score}\nConfusion Matrix:\n{confusion_matrix}")

def training_plot(accuracy, loss, model_name):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.plot(accuracy, color="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss")
    ax2.plot(loss, color="tab:orange")

    fig.legend(['Accuracy', 'Loss'], loc='upper left')
    plt.savefig(f'graphs/{model_name}.png')