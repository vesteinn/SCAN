train_file=../data/SCAN/length_split/tasks_train_length.txt
valid_file=../data/SCAN/length_split/tasks_test_length.txt

#DEBUG MODE
train_file=../data/SCAN/simple_split/tasks_train_simple.txt
valid_file=../data/SCAN/simple_split/tasks_test_simple.txt

#for i in {0..4}; do
for i in {0..0}; do
    log_dir=../logs_transformer/experiment_2/
    mkdir -p $log_dir
    log_file=$log_dir/log_tf_${i}.txt
    echo "Starting run ${i} for Transformer model"
    python ../src/scan/train.py --device cpu \
        --train $train_file --valid $valid_file --model transformer \
        --dropout 0.5 \
        --nheads 4 --lr 0.001 --layers 1 --hidden_dim 100 --eval_interval 500 --verbose #> debugout3_for_paper #--verbose > $log_file
    tail -n 1 $log_file
    echo "--"
done


# --use_oracle \


