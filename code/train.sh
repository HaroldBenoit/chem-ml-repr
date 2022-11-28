#!/bin/bash

for name in alpha homo lumo gap r2 zpve cv h298 
do 
    python3 training.py --target $name --cluster
done
