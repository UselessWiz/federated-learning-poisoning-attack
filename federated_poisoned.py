import numpy as np
import tensorflow as tf
import random

import shared

train_data, train_label, test_data, test_label = shared.process_data()
split_train_data, split_train_label = shared.split_train(train_data, train_label, shared.client_count)

attacker = random.randint(0, shared.client_count - 1) 

for i in range(len(split_train_label[attacker])):
    split_train_label[attacker][i] = int(np.where(shared.label_encoder.classes_ == 'Normal')[0])

train_datasets = []

for i in range(0, len(split_train_data)):
    train_datasets.append(tf.data.Dataset.from_tensors((split_train_data[i], split_train_label[i])))

test_dataset = tf.data.Dataset.from_tensors((test_data, test_label))

input_spec = train_datasets[0].element_spec

model = shared.build_model((input_spec[0].shape))
trainer = shared.tff_setup(model, input_spec, shared.learning_rate)

# Run the federated learning process over a number of rounds. 
# Keep track of the state with the highest training accuracy.
max_accuracy_result, accuracy, loss = shared.federated_train(trainer, train_datasets, shared.round_count)

shared.training_plot(accuracy, loss, "federated_poisoned")

shared.federated_evaluation(input_spec, max_accuracy_result, test_dataset, test_data, test_label, "FEDERATED POISONED")