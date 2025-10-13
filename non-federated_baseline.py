import pandas as pd
import tensorflow as tf
import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

def build_model():
    # MODEL PARAMETERS
    hidden_layer_count =    10
    neuron_count =          10
    activation =            'relu'
    optimizer =             'sgd'
    epochs =                10000

    model = keras.models.Sequential()
    #model.add(keras.Input())
    for _ in range(hidden_layer_count):
        model.add(keras.layers.Dense(neuron_count, activation=activation))
    model.add(keras.layers.Dense(6, activation='softmax'))

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_dataset, epochs=epochs)

    # Evaluate the model
    train_evaluation = model.evaluate(train_dataset)
    test_evaluation = model.evaluate(test_dataset)
    return model, history, train_evaluation, test_evaluation

# Prepare Data

train_data = pd.read_csv('train.csv')
train_label = train_data['Label']
train_data = train_data.drop('Label', axis=1)
test_data = pd.read_csv('test.csv')
test_label = test_data['Label']
test_data = test_data.drop('Label', axis=1)

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

train_dataset = tf.data.Dataset.from_tensors((train_data, train_label))
test_dataset = tf.data.Dataset.from_tensors((test_data, test_label))

model, history, train_evaluation, test_evalutaion = build_model()

print(train_evaluation)

prediction = model.predict(test_data).argmax(axis=1)
accuracy_score = accuracy_score(test_label, prediction)
confusion_matrix = confusion_matrix(test_label, prediction)
print(f"Testing - Prediction Accuracy: {accuracy_score}\nConfusion Matrix:\n{confusion_matrix}")