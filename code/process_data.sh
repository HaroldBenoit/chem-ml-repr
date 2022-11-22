#!/bin/bash

read -p "Please enter dataset name: " dataset
read -p "Please enter root directory: " root

python3 datasets.py --dataset $dataset --root $root