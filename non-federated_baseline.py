import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix

import shared

train_data, train_label, test_data, test_label = shared.process_data()

train_dataset = tf.data.Dataset.from_tensors((train_data, train_label))
test_dataset = tf.data.Dataset.from_tensors((test_data, test_label))

input_spec = train_data.shape

model = shared.build_model(input_spec)

history = model.fit(train_dataset, epochs=shared.round_count)

# Evaluate the model
def evaluate(train_dataset, test_dataset):
    train_evaluation = model.evaluate(train_dataset)
    test_evaluation = model.evaluate(test_dataset)
    return train_evaluation, test_evaluation

train_evaluation, test_evalutaion = evaluate(train_dataset, test_dataset)

prediction = model.predict(test_data).argmax(axis=1)
accuracy_score = accuracy_score(test_label, prediction)
confusion_matrix = confusion_matrix(test_label, prediction)
print(f"Testing - Prediction Accuracy: {accuracy_score}\nConfusion Matrix:\n{confusion_matrix}")