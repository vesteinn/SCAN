train_file=../../data/SCAN/simple_split/tasks_train_simple.txt
valid_file=../../data/SCAN/simple_split/tasks_test_simple.txt

#EMB_SIZE = sys.argv[3] #100
#NHEAD = 4
#FFN_HID_DIM = sys.argv[3] #100
#NUM_ENCODER_LAYERS = sys.argv[4] #1
#NUM_DECODER_LAYERS = sys.argv[4] #1

dim=200
layers=2
bsz=1
lr=0.0001

for i in {0..4}; do
    log_dir=../logs_alttransformer/experiment_1/dim${dim}_layers${layers}_bsz${bsz}_lr${lr}
    mkdir -p $log_dir
    log_file=$log_dir/log_${i}.txt
    echo "Starting run ${i} for transformer model"
    python ../../src/scan/alttransformer.py ${train_file} ${valid_file} ${dim} ${layers} ${bsz} ${i} ${lr}> $log_file
    tail -n 1 $log_file
    echo "--"
done




