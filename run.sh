#!/bin/bash

python3 non-federated_baseline.py
python3 federated_baseline.py
python3 federated_poisoned.py
python3 federated_poisoned_submodel.py
python3 federated_poisoned_zeroed.py
python3 federated_poisoned_rfa.py