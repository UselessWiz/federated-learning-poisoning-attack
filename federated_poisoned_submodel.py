import numpy as np
import tensorflow as tf
import random
import time

import shared

train_data, train_label, test_data, test_label = shared.process_data()
split_train_data, split_train_label = shared.split_train(train_data, train_label, shared.client_count)

"""
One of the clients has poisoned training data with the intention of manipulating the model
to allow all attacks through. It does this by setting the label of all data to "Normal"

We test this by looking for a difference in accuracy and in the confusion matrix; there should
be more false negatives.
"""

# The known good source is list index 0, so we choose an attacker at a different index.
attacker = random.randint(1, shared.client_count - 1) 

for i in range(len(split_train_label[attacker])):
    split_train_label[attacker][i] = int(np.where(shared.label_encoder.classes_ == 'Normal')[0])

train_datasets = []

for i in range(1, len(split_train_data)):
    train_datasets.append(tf.data.Dataset.from_tensors((split_train_data[i], split_train_label[i])))

test_dataset = tf.data.Dataset.from_tensors((test_data, test_label))

input_spec = train_datasets[0].element_spec

model = shared.build_model((input_spec[0].shape))

# Train the base model using a trusted dataset (the first dataset, which we ensure cannot be the attacker).
trusted_dataset = tf.data.Dataset.from_tensors((split_train_data[0], split_train_label[0]))

history = model.fit(trusted_dataset, epochs=5000)
train_evaluation = model.evaluate(trusted_dataset) # Provide confirmation that the trusted model has trained appropriately.

trainer = shared.tff_setup(model, input_spec, shared.learning_rate)

# Run the federated learning process over a number of rounds. 
# Keep track of the state with the highest training accuracy.
max_accuracy_result, accuracy, loss = shared.federated_train(trainer, train_datasets, shared.round_count)

shared.training_plot(accuracy, loss, "federated_poisoned_submodel")

shared.federated_evaluation(input_spec, max_accuracy_result, test_dataset, test_data, test_label, "FEDERATED POISONED - SUBMODEL PRE-TRAINING")

