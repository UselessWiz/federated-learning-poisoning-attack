# Federated Learning + Poisoning Attack

## Results

### Non-Federated (Control)
This version has a s
```
Best Accuracy: 0.777011513710022 (Training evaluation Accuracy, not necessarily best)
1/1 [==============================] - 0s 104ms/step - loss: 0.7369 - accuracy: 0.7184
18/18 [==============================] - 0s 544us/step
Testing - Prediction Accuracy: 0.7183600713012478
Confusion Matrix:
[[ 89  26   0   0   0   7]
 [ 29  56   0   0   1   4]
 [  0   1  48   0   0   0]
 [  5   0   0   2   1   9]
 [  1   2   0   0  79  45]
 [  1   0   0   1  25 129]]
 ```

### Federated Learning
```
Best Accuracy: 0.8513410091400146
1/1 [==============================] - 0s 149ms/step - loss: 0.8577 - accuracy: 0.7487
18/18 [==============================] - 0s 529us/step
Testing - Prediction Accuracy: 0.7486631016042781
Confusion Matrix:
[[ 85  31   0   2   2   2]
 [ 17  71   0   0   0   2]
 [  0   0  48   0   1   0]
 [  5   0   0   3   3   6]
 [  6   0   3   4  78  36]
 [  0   0   0   1  20 135]]
```

### Federated Learning when Poisoned
```
Best Accuracy: 0.7222222089767456
1/1 [==============================] - 0s 147ms/step - loss: 1.1620 - accuracy: 0.6471
18/18 [==============================] - 0s 531us/step
Testing - Prediction Accuracy: 0.6470588235294118
Confusion Matrix:
[[ 52  45  16   5   0   4]
 [  9  74   5   0   1   1]
 [  0   0  49   0   0   0]
 [  1   1   5   2   0   8]
 [  0   4  11   3  64  45]
 [  0   0   6   2  26 122]]
```

### First pass Federated Learning Poisoning Mitigation (Submodel)
This works by creating a model and training it on a small set of trusted data.
This model is then used by all clients, training on their subset (including the one attacker).

This tends to have a higher training accuracy, but falls short when making predictions, in some cases performing worse than the poisoned model on it's own.
(THE PROVIDED RESULTS ARE AN OUTLIER FROM WHAT NORMALLY APPEARS...)
```
Best Accuracy: 0.7054597735404968
1/1 [==============================] - 0s 147ms/step - loss: 1.5152 - accuracy: 0.6988
18/18 [==============================] - 0s 804us/step
Testing - Prediction Accuracy: 0.6987522281639929
Confusion Matrix:
[[ 68  42   6   1   2   3]
 [ 32  51   3   2   0   2]
 [  0   0  49   0   0   0]
 [  2   2   1   3   5   4]
 [  2   4   2   2  85  32]
 [  0   0   4   1  15 136]]
```

### Federated Learning Poisoning Mitigation (Zeroing Aggregation)
According to [this](https://www.tensorflow.org/federated/tutorials/tuning_recommended_aggregators) from the TFF docs, zeroing can help increase resiliance to "data corruption from faulty clients", or poisoning attacks authored by malicious federated learning parties.

In testing, it's done little to improve results, and actually creates more false negatives than other models.
```
Best Accuracy: 0.7448275685310364
1/1 [==============================] - 0s 154ms/step - loss: 1.2580 - accuracy: 0.6542
18/18 [==============================] - 0s 520us/step
Testing - Prediction Accuracy: 0.6541889483065954
Confusion Matrix:
[[85 24  8  2  1  2]
 [18 65  5  0  1  1]
 [ 0  0 49  0  0  0]
 [ 4  1  2  2  1  7]
 [10  3 12  1 67 34]
 [ 1  2 20  0 34 99]]

Best Accuracy: 0.7015325427055359
1/1 [==============================] - 0s 158ms/step - loss: 1.2719 - accuracy: 0.6738
18/18 [==============================] - 0s 547us/step
Testing - Prediction Accuracy: 0.6737967914438503
Confusion Matrix:
[[ 74  39   3   2   0   4]
 [ 23  61   2   1   0   3]
 [  0   0  49   0   0   0]
 [  3   0   1   5   0   8]
 [  0   0  17   6  58  46]
 [  2   0   0   4  19 131]]
```

### Federated Learning Poisoning Mitigation (Robust Federated Aggregation)
According to the paper, this is designed to perform better than FedAvg by using the Geometric Median. In their experimentation
```
Best Accuracy: 0.7371647357940674
1/1 [==============================] - 0s 153ms/step - loss: 1.2159 - accuracy: 0.7130
18/18 [==============================] - 0s 528us/step
FEDERATED POISONED - KRUM
Testing - Prediction Accuracy: 0.7130124777183601
Confusion Matrix:
[[ 74  42   0   2   0   4]
 [ 26  62   0   0   1   1]
 [  0   0  47   0   2   0]
 [  3   1   0   4   5   4]
 [  0   7   1   4  86  29]
 [  0   1   0   2  26 127]]
```