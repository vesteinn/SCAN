train_file=../data/SCAN/length_split/tasks_train_length.txt
valid_file=../data/SCAN/length_split/tasks_test_length.txt



#for i in {0..4}; do
for i in {0..0}; do
    log_dir=../logs_transformer/experiment_2/
    mkdir -p $log_dir
    log_file=$log_dir/log_tf_${i}.txt
    echo "Starting run ${i} for Transformer model"
    python ../src/scan/train.py --device cpu \
        --train $train_file --valid $valid_file --model transformer \
        --nheads 4 --hidden_dim 128 --eval_interval 100 --verbose #> $log_file
    tail -n 1 $log_file
    echo "--"
done




