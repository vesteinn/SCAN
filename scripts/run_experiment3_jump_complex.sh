#
# jump primitive
# 
train_file=../data/SCAN/add_prim_split/with_additional_examples/tasks_train_addprim_complex_jump
valid_file=../data/SCAN/add_prim_split/with_additional_examples/tasks_test_addprim_complex_jump


for split in 1 2 4 8 16 32; do
	for i in {1..5}; do
		log_dir=../logs/experiment_3/jump_complex
		mkdir -p $log_dir
		log_file=$log_dir/log_p${split}_${i}.txt
		model_name=$log_dir/model_p${split}_${i}.pt
		echo "Starting run ${i} for best performing model"
		python ../src/scan/train.py --device cuda \
			--use_attention \
			--dropout 0.1 \
			--verbose \
			--name ${model_name} \
			--log_target_probs \
			--train ${train_file}_num${split}_rep${i}.txt \
			--valid ${valid_file}_num${split}_rep${i}.txt \
			--model lstm \
			--layers 1 --hidden_dim 100 --eval_interval 105000 > $log_file
		tail -n 1 $log_file
		echo "--"
	done
done

