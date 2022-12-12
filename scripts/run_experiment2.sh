train_file=../data/SCAN/length_split/tasks_train_length.txt
valid_file=../data/SCAN/length_split/tasks_test_length.txt


for i in {0..4}; do
    log_dir=../logs/experiment_2
    mkdir -p $log_dir
    log_file=$log_dir/log_${i}.txt
    echo "Starting run ${i} for best performing model"
    python ../src/scan/train.py --device cuda \
         --use_attention \
	 --log_target_probs \
	 --use_oracle \
         --train $train_file --valid $valid_file --model gru \
         --layers 1 --hidden_dim 50 --eval_interval 105000 > $log_file
    tail -n 1 $log_file
    echo "--"
done



for i in {0..4}; do
    log_dir=../logs/experiment_2/ob
    mkdir -p $log_dir
    log_file=$log_dir/log_ob_${i}.txt
    echo "Starting run ${i} for overall best model"
    python ../src/scan/train.py --device cuda \
	--use_oracle \
        --train $train_file --valid $valid_file --model lstm \
	--log_target_probs \
        --layers 2 --hidden_dim 100 --eval_interval 105000 > $log_file
    tail -n 1 $log_file
    echo "--"
done




