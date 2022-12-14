
for i in {0..4}; do
    log_dir=../logs/experiment_1/ob
    mkdir -p $log_dir
	    log_file=$log_dir/log_ob_${i}.txt
    echo "Starting run ${i} for overall best model"
    python ../src/scan/train.py --device cuda --eval_interval 5000 --train ../data/SCAN/simple_split/tasks_train_simple.txt --valid ../data/SCAN/simple_split/tasks_test_simple.txt --model lstm > $log_file
    tail -n 1 $log_file
    echo "--"
done

for i in {0..4}; do
    log_dir=../logs/experiment_1
    mkdir -p $log_dir
    log_file=$log_dir/log_top_perf_${i}.txt
    echo "Starting run ${i} for best performing model"
    python ../src/scan/train.py --device cuda --eval_interval 101000 --train ../data/SCAN/simple_split/tasks_train_simple.txt --valid ../data/SCAN/simple_split/tasks_test_simple.txt --model lstm --dropout 0 > $log_file
    tail -n 1 $log_file
    echo "--"
done

for split in p1 p2 p4 p8 p16 p32 p64; do
    for i in {0..4}; do
        train_file=../data/SCAN/simple_split/size_variations/tasks_train_simple_${split}.txt
        valid_file=../data/SCAN/simple_split/size_variations/tasks_test_simple_${split}.txt
        log_dir=../logs/experiment_1/split_variations
        mkdir -p $log_dir
        log_file=$log_dir/log_ob_${split}_${i}.txt
        echo "Starting run ${i} for overall best performing model on split ${split}"
        python ../src/scan/train.py --device cuda --eval_interval 101000 --train ${train_file} --valid ${valid_file} --model lstm > $log_file
        tail -n 1 $log_file
        echo "--"
    done
done


