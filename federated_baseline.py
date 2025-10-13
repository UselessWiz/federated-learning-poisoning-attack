import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import keras
from collections import OrderedDict
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

train_data = pd.read_csv('train.csv')
train_data = train_data.sample(frac=1).reset_index(drop=True) # Shuffle data.
train_label = train_data['Label']
train_data = train_data.drop('Label', axis=1)
test_data = pd.read_csv('test.csv')
test_label = test_data['Label']
test_data = test_data.drop('Label', axis=1)

# OVERALL PARAMETERS
client_count        = 5     # number of clients
round_count         = 10000 # Number of rounds each client runs
learning_rate       = 0.1   # Model learning rate.

# MODEL PARAMETERS
hidden_layer_count  = 10
neuron_count        = 10
activation          = 'relu'
optimizer           = 'sgd'

def unison_shuffled_copies(a, b):
    """
    Helper function for shuffling train and test arrays in unison.
    
    Source: https://stackoverflow.com/a/4602224
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

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

# Convert data to all numeric types, using one-hot encoding on categorical data
train_data = pd.get_dummies(train_data, drop_first=True)
test_data = pd.get_dummies(test_data, drop_first=True)

# Scale all data
scaler = StandardScaler()
train_data = pd.DataFrame(scaler.fit_transform(train_data), columns = train_data.columns)
test_data = pd.DataFrame(scaler.transform(test_data), columns = test_data.columns)

# Encode labels to numbers
label_encoder = LabelEncoder()
train_label = label_encoder.fit_transform(train_label)
test_label = label_encoder.transform(test_label)

# Split training data between clients.
split_train_data = np.array_split(train_data, client_count)
split_train_label = np.array_split(train_label, client_count)

train_datasets = []

for i in range(len(split_train_data)):
    train_datasets.append(tf.data.Dataset.from_tensors((split_train_data[i], split_train_label[i])))

test_dataset = tf.data.Dataset.from_tensors((test_data, test_label))

# Create a base model
model = build_model((train_datasets[0].element_spec[0].shape))

# Make the model federated-learning compatible for tff to use
model = tff.learning.models.functional_model_from_keras(
    model, keras.losses.SparseCategoricalCrossentropy(), 
    train_datasets[0].element_spec,
    metrics_constructor=OrderedDict(accuracy=keras.metrics.SparseCategoricalAccuracy)
)

# Create the federated learning training setup
trainer = tff.learning.algorithms.build_weighted_fed_avg(
   model, 
   client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=learning_rate)
)

# Run the federated learning process over a number of rounds. 
# Keep track of the state with the highest training accuracy.
state = trainer.initialize()
max_accuracy_result = None
for i in range(round_count):
    result = trainer.next(state, train_datasets)
    state = result.state
    metrics = result.metrics
    if max_accuracy_result is None or \
      metrics['client_work']['train']['accuracy'] >= max_accuracy_result.metrics['client_work']['train']['accuracy']:
        max_accuracy_result = result
    print(f"Round {i} Accuracy: {metrics['client_work']['train']['accuracy']}")

# Prepare an inference model based on the best performing state.
# This will be used to predict based on the test data we set aside.
inference_model = build_model((test_dataset.element_spec[0].shape))
print(f"Best Accuracy: {max_accuracy_result.metrics['client_work']['train']['accuracy']}")
max_accuracy_result.state.global_model_weights.assign_weights_to(inference_model)

# Evaluate the model's performance on the test dataset.
evaluation = inference_model.evaluate(test_dataset)

# Run some predictions on the model and generate a confusion matrix
prediction = inference_model.predict(test_data).argmax(axis=1)
accuracy_score = accuracy_score(test_label, prediction)
confusion_matrix = confusion_matrix(test_label, prediction)
print("FEDERATED BASELINE")
print(f"Testing - Prediction Accuracy: {accuracy_score}\nConfusion Matrix:\n{confusion_matrix}")

