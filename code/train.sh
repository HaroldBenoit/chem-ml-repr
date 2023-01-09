#!/bin/bash

#muv = MUV-466 MUV-548 MUV-600 MUV-644 MUV-652 MUV-689 MUV-692 MUV-712 MUV-713 MUV-733 MUV-737 MUV-810 MUV-832 MUV-846 MUV-852 MUV-858 MUV-859

#qm9 = mu alpha  homo  lumo  gap  r2  zpve  cv u0  u298 h298 g298

#for name in zpve  cv u0  u298 h298 g298
#do
#    python3 training.py --target $name --dataset qm9  --root ../data/qm9  --cluster --hydrogen
#done
#
#for name in mu alpha  homo  lumo  gap  r2  zpve  cv u0  u298 h298 g298
#do
#    python3 training.py --target $name --dataset qm9  --root ../data/qm9  --cluster --hydrogen --weighted
#done

echo STARTING

python3 training.py --root ../data/bace --dataset bace --target Class --cluster --epochs 50 --hydrogen --num_message_layers 2 --seed 100

sleep 60s

python3 training.py --root ../data/bace --dataset bace --target pIC50 --cluster --epochs 50 --hydrogen --num_message_layers 2 --seed 100

sleep 60s

python3 training.py --root ../data/bbbp --dataset bbbp --target p_np --cluster --epochs 50 --hydrogen --num_message_layers 2 --seed 100

sleep 60s

python3 training.py --root ../data/freesolv --dataset freesolv --target y --cluster --epochs 50 --hydrogen --num_message_layers 2 --seed 100

sleep 60s


#sleep 2h

for name in gap mu alpha  homo  lumo r2  zpve  cv u0  u298 h298 g298
do
    python3 training.py --target $name --dataset qm9  --root ../data/qm9  --cluster --hydrogen --epochs 4 --batch_size 16 --num_message_layers 2 --val_check_interval 0.25 --seed 100
    sleep 60s

done

