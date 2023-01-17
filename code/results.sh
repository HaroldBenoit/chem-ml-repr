mkdir ../experiments_results
touch ../experiments_results/global_results.csv


for i in "bbbp p_np" "freesolv y" "bace Class" "bace pIC50" "lipo exp" 
do
    set -- $i # converts the "tuple" into the pram args $1 $2
    for seed in 100 200 300
        do

        python3 training.py --no_log --results --hydrogen --cluster --root ../data/$1 --target $2 --dataset $1 --train_frac 0.8 --valid_frac 0.2 --test_frac 0.0 --seed $seed --num_message_layers 3 --scaffold --epochs 100
        sleep 15s

        done

done




for i in "phonons phonons" "dieletric n" "log_gvrh log_gvrh" "perovskites e_form" 
do
    set -- $i # converts the "tuple" into the pram args $1 $2
    for seed in 100 200 300
        do

        python3 training.py --no_log --results --cluster --root ../data/matbench/$1 --target $2 --dataset $1 --train_frac 0.8 --valid_frac 0.2 --test_frac 0.0 --seed $seed --num_message_layers 3 --epochs 100 --boolean
        sleep 15s

        done

done





### 3D randomness checking
#
#for i in "bbbp p_np" "freesolv y" "bace Class" "bace pIC50" "lipo exp" 
#do
#    set -- $i # converts the "tuple" into the pram args $1 $2
#    for seed in 100 200 300
#        do
#
#        rm -rf ../data/$1*/processed
#
#        python3 datasets_classes.py --root ../data/$1 --hydrogen --seed $seed --dataset $1
#
#        python3 training.py --no_log --results --hydrogen --cluster --root ../data/$1 --target $2 --dataset $1 --train_frac 0.8 --valid_frac 0.2 --test_frac 0.0 --seed 100 --num_message_layers 3 --scaffold --epochs 100
#        sleep 15s
#
#        done
#
#done
