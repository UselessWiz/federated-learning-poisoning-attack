import numpy as np
import tensorflow as tf
import random
import tensorflow_federated as tff
import keras
import collections

import shared
import rfa_factory

train_data, train_label, test_data, test_label = shared.process_data()
split_train_data, split_train_label = shared.split_train(train_data, train_label, shared.client_count)

"""
One of the clients has poisoned training data with the intention of manipulating the model
to allow all attacks through. It does this by setting the label of all data to "Normal"

We test this by looking for a difference in accuracy and in the confusion matrix; there should
be more false negatives.
"""

attacker = random.randint(0, shared.client_count - 1) 

for i in range(len(split_train_label[attacker])):
    split_train_label[attacker][i] = int(np.where(shared.label_encoder.classes_ == 'Normal')[0])

train_datasets = []

for i in range(0, len(split_train_data)):
    train_datasets.append(tf.data.Dataset.from_tensors((split_train_data[i], split_train_label[i])))

test_dataset = tf.data.Dataset.from_tensors((test_data, test_label))

input_spec = train_datasets[0].element_spec

model = shared.build_model((input_spec[0].shape))
# Make the model federated-learning compatible for tff to use
model = tff.learning.models.functional_model_from_keras(
    model, keras.losses.SparseCategoricalCrossentropy(), 
    input_spec,
    metrics_constructor=collections.OrderedDict(accuracy=keras.metrics.SparseCategoricalAccuracy)
)

trainer = tff.learning.algorithms.build_weighted_fed_avg(
    model,
    client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=shared.learning_rate),
    model_aggregator=rfa_factory.RobustWeiszfeldFactory()
)

# Run the federated learning process over a number of rounds. 
# Keep track of the state with the highest training accuracy.
max_accuracy_result = shared.federated_train(trainer, train_datasets, shared.round_count)

shared.federated_evaluation(input_spec, max_accuracy_result, test_dataset, test_data, test_label, "FEDERATED POISONED ZEROED")

