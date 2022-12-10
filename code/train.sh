#!/bin/bash

#muv = MUV-466 MUV-548 MUV-600 MUV-644 MUV-652 MUV-689 MUV-692 MUV-712 MUV-713 MUV-733 MUV-737 MUV-810 MUV-832 MUV-846 MUV-852 MUV-858 MUV-859

#qm9 = mu alpha  homo  lumo  gap  r2  zpve  cv u0  u298 h298 g298

for name in zpve  cv u0  u298 h298 g298
do
    python3 training.py --target $name --dataset qm9  --root ../data/qm9  --cluster --hydrogen
done

for name in mu alpha  homo  lumo  gap  r2  zpve  cv u0  u298 h298 g298
do
    python3 training.py --target $name --dataset qm9  --root ../data/qm9  --cluster --hydrogen --weighted
done
