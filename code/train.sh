#!/bin/bash

for name in MUV-466 MUV-548 MUV-600 MUV-644 MUV-652 MUV-689 MUV-692 MUV-712 MUV-713 MUV-733 MUV-737 MUV-810 MUV-832 MUV-846 MUV-852 MUV-858 MUV-859
do 
    python3 training.py --target $name --dataset muv  --root ../data/muv  --cluster
done
