import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import keras
from collections import OrderedDict
import random

import shared

"""
KRUM is an alternate method of aggregating the results of all the clients.
This has been custom developed to work with TFF.

NOTE THAT THIS IMPLEMENTATION IS NOT TRULY PRIVATE, AND IS CONSTRUCTED PURELY 
FOR DEMONSTRATION PURPOSES.
"""

client_count = 5

# Implementing this requires the creation of a new Federated Learning algorithm inside TFF.

train_data, train_label, test_data, test_label = shared.process_data()
split_train_data, split_train_label = shared.split_train(train_data, train_label, client_count)

attacker = random.randint(0, client_count - 1) 

for i in range(len(split_train_label[attacker])):
    split_train_label[attacker][i] = int(np.where(shared.label_encoder.classes_ == 'Normal')[0])

train_datasets = []

for i in range(0, len(split_train_data)):
    train_datasets.append(tf.data.Dataset.from_tensors((split_train_data[i], split_train_label[i])))

test_dataset = tf.data.Dataset.from_tensors((test_data, test_label))

# Create a base model
model = shared.build_model((train_datasets[0].element_spec[0].shape))

tff_model = tff.learning.models.functional_model_from_keras(
    model, keras.losses.SparseCategoricalCrossentropy(), 
    train_datasets[0].element_spec,
    metrics_constructor=OrderedDict(accuracy=keras.metrics.SparseCategoricalAccuracy)
)

@tf.py_function(Tout=(tf.int32))
def calculate_krum(client_weights):
    def euclidean_distance(x, y):
        return np.linalg.norm(x - y)
    
    numpy_weights = [list(weight.convert_variables_to_arrays()) for weight in client_weights]

    def krum(weights):
        num_clients = len(weights)
        dist_matrix = np.zeros((num_clients, num_clients))

        # Calculate the distance between weights
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = euclidean_distance(weights[i], weights[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        # Calculate the sum of distances for each participant and select the model with the minimum sum of distances
        min_sum_dist = float('inf')
        selected_index = -1
        for i in range(num_clients):
            sorted_indices = np.argsort(dist_matrix[i])
            sum_dist = np.sum(dist_matrix[i, sorted_indices[1:(num_clients - 1)]])
            if sum_dist < min_sum_dist:
                min_sum_dist = sum_dist
                selected_index = i

        return selected_index
    
    return krum(numpy_weights)


@tf.function
def client_update(model, dataset, initial_weights, client_optimizer):
    """Performs training (using the server model weights) on the client's dataset."""
    # Initialize the client model with the current server weights and the optimizer
    # state.
    client_weights = initial_weights.trainable
    optimizer_state = client_optimizer.initialize(
        tf.nest.map_structure(tf.TensorSpec.from_tensor, client_weights)
    )

    weights = []

    # Use the client_optimizer to update the local model.
    # In this setup, while there is only one TFF client, 
    # each client's work is represented by the batch inside of the dataset.
    for batch in dataset:
        x, y = batch
        with tf.GradientTape() as tape:
            tape.watch(client_weights)
            # Compute a forward pass on the batch of data
            outputs = model.predict_on_batch(
                model_weights=(client_weights, ()), x=x, training=True
            )
            loss = model.loss(output=outputs, label=y)

        # Compute the corresponding gradient
        grads = tape.gradient(loss, client_weights)

        # Apply the gradient using a client optimizer.
        optimizer_state, client_weights = client_optimizer.next(
            optimizer_state, weights=client_weights, gradients=grads
        )

        weights.append(client_weights)
    
    best_model_weights = tff.learning.models.ModelWeights(weights[int(calculate_krum(weights))], non_trainable=())
    return best_model_weights

    #return tff.learning.models.ModelWeights(client_weights, non_trainable=())

@tff.tensorflow.computation
def server_init():
    return tff.learning.models.ModelWeights(*tff_model.initial_weights)

@tff.federated_computation()
def initialize_fn():
    return tff.federated_eval(
        server_init, tff.SERVER
    )

@tf.function
def server_update(model, best_client_weights):
    """Updates the server model weights as the average of the client model weights."""
    del model  # Unused, just take the best_client_weights.
    return best_client_weights

tf_dataset_type = tff.SequenceType(
    tff.types.tensorflow_to_type(tff_model.input_spec)
)

model_weights_type = server_init.type_signature.result

@tff.tensorflow.computation(tf_dataset_type, model_weights_type)
def client_update_fn(tf_dataset, server_weights):
  client_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=0.01)
  return client_update(tff_model, tf_dataset, server_weights, client_optimizer)

@tff.tensorflow.computation(model_weights_type)
def server_update_fn(best_client_weights):
  return server_update(tff_model, best_client_weights)

federated_server_type = tff.FederatedType(
    model_weights_type, tff.SERVER
)
federated_dataset_type = tff.FederatedType(
    tf_dataset_type, tff.CLIENTS
)

@tff.federated_computation(
    federated_server_type, federated_dataset_type
)
def next_fn(server_weights, federated_dataset):
    # Broadcast the server weights to the clients.
    server_weights_at_client = tff.federated_broadcast(
          server_weights
    )

    # We have already calculated the best weights.
    # This is not the correct way to do this at all; in this implementation 
    # one client technically can see all the data. The way it's using it however 
    # makes the algorithm identical to an actual application of KRUM.
    # It's been stated before, but do not use these ideas for real applications.
    best_client_weights = tff.federated_map(
        client_update_fn, (federated_dataset, server_weights_at_client)
    )

    # The server updates its model.
    server_weights = tff.federated_map(
        server_update_fn, best_client_weights
    )

    return server_weights

federated_algorithm = tff.templates.IterativeProcess(
    initialize_fn=initialize_fn, next_fn=next_fn
)

def evaluate(model_weights):
    keras_model = shared.build_model((train_datasets[0].element_spec[0].shape))
    keras_model.compile(
          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model_weights.assign_weights_to(keras_model)
    keras_model.evaluate(test_dataset)

server_state = federated_algorithm.initialize()
evaluate(server_state)

round_count = 5000

max_accuracy_result = None
for i in range(round_count):
    server_state = federated_algorithm.next(server_state, train_datasets)
    #state = result.state
    #print(result)
    """
    if max_accuracy_result is None or \
      metrics['client_work']['train']['accuracy'] >= max_accuracy_result.metrics['client_work']['train']['accuracy']:
        max_accuracy_result = result
    print(f"Round {i} Accuracy: {metrics['client_work']['train']['accuracy']}")"""

#for i in range(20):
#    server_state = federated_algorithm.next(server_state, train_datasets)

evaluate(server_state)