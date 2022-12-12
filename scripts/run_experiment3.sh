#
# turn left primitive
# 
train_file=../data/SCAN/add_prim_split/tasks_train_addprim_turn_left.txt
valid_file=../data/SCAN/add_prim_split/tasks_test_addprim_turn_left.txt


for i in {0..4}; do
    log_dir=../logs/experiment_3/turn_left
    mkdir -p $log_dir
    log_file=$log_dir/log_${i}.txt
    model_name=$log_dir/model_${i}.pt
    echo "Starting run ${i} for best performing model"
    python ../src/scan/train.py --device cuda \
         --use_attention \
         --dropout 0.1 \
	 --verbose \
	 --train $train_file --valid $valid_file --model gru \
	 --name ${model_name} \
         --layers 1 --hidden_dim 100 --eval_interval 105000 > $log_file
    tail -n 1 $log_file
    echo "--"
done



for i in {0..4}; do
    log_dir=../logs/experiment_3/turn_left/ob
    mkdir -p $log_dir
    log_file=$log_dir/log_ob_${i}.txt
    model_name=$log_dir/model_${i}.pt
    echo "Starting run ${i} for overall best model"
    python ../src/scan/train.py --device cuda \
	--use_oracle \
	--verbose \
	--name ${model_name} \
        --train $train_file --valid $valid_file --model lstm \
	--log_target_probs \
        --layers 2 --hidden_dim 100 --eval_interval 105000 > $log_file
    tail -n 1 $log_file
    echo "--"
done


#
# jump primitive
# 
train_file=../data/SCAN/add_prim_split/tasks_train_addprim_jump.txt
valid_file=../data/SCAN/add_prim_split/tasks_test_addprim_jump.txt


for i in {0..4}; do
    log_dir=../logs/experiment_3/jump
    mkdir -p $log_dir
    log_file=$log_dir/log_${i}.txt
    model_name=$log_dir/model_${i}.pt
    echo "Starting run ${i} for best performing model"
    python ../src/scan/train.py --device cuda \
         --use_attention \
         --dropout 0.1 \
	 --verbose \
	 --name ${model_name} \
	 --log_target_probs \
	 --train $train_file --valid $valid_file --model lstm \
         --layers 1 --hidden_dim 100 --eval_interval 105000 > $log_file
    tail -n 1 $log_file
    echo "--"
done



for i in {0..4}; do
    log_dir=../logs/experiment_3/jump/ob
    mkdir -p $log_dir
    model_name=$log_dir/model_${i}.pt
    log_file=$log_dir/log_ob_${i}.txt
    echo "Starting run ${i} for overall best model"
    python ../src/scan/train.py --device cuda \
	--use_oracle \
	--verbose \
        --train $train_file --valid $valid_file --model lstm \
	--name ${model_name} \
	--log_target_probs \
        --layers 2 --hidden_dim 100 --eval_interval 101000 > $log_file
    tail -n 1 $log_file
    echo "--"
done



