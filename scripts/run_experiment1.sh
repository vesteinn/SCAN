for i in {0..5}; do
    log_dir=../data/logs/experiment_${i}/ob
    mkdir -p $log_dir
    echo "Starting run ${i} for overall best model"
    python ../src/scan/train.py --train ../data/SCAN/simple_split/tasks_train_simple.txt --valid ../data/SCAN/simple_split/tasks_test_simple.txt --model lstm > $log_dir/
done

for i in {0..5}; do
    log_dir=../data/logs/experiment_${i}
    mkdir -p $log_dir
    log_file=$log_dir/log_top_perf_${i}.txt
    echo "Starting run ${i} for best performing model"
    python ../src/scan/train.py --train ../data/SCAN/simple_split/tasks_train_simple.txt --valid ../data/SCAN/simple_split/tasks_test_simple.txt --model lstm --dropout 0 > $log_file
    tail -n 1 $log_file
    echo "--"
done
