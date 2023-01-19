echo "Starting debug run ${i} for overall best model"
python ../src/scan/train.py --device cpu --eval_interval 500 --train ../data/SCAN/simple_split/tasks_train_simple.txt \
--valid ../data/SCAN/simple_split/tasks_test_simple.txt --model lstm --verbose

