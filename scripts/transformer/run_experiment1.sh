for split in p1 p2 p4 p8 p16 p32 p64; do
    for i in {0..4}; do
        train_file=../data/SCAN/simple_split/size_variations/tasks_train_simple_${split}.txt
        valid_file=../data/SCAN/simple_split/size_variations/tasks_test_simple_${split}.txt
        log_dir=../logs_transformer/experiment_1/split_variations
        mkdir -p $log_dir
        log_file=$log_dir/log_tf_${split}_${i}.txt
        echo "Starting run ${i} for Transformer on split ${split}"
	python ../src/scan/train.py --device cpu --eval_interval 10000 --train ${train_file} --valid ${valid_file} --nheads 4 --hidden_dim 128 --model transformer > $log_file
        tail -n 1 $log_file
        echo "--"
    done
done


