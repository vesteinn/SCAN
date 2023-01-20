train_file=../../data/SCAN/length_split/tasks_train_length.txt
valid_file=../../data/SCAN/length_split/tasks_test_length.txt

#EMB_SIZE = sys.argv[3] #100
#NHEAD = 4
#FFN_HID_DIM = sys.argv[3] #100
#NUM_ENCODER_LAYERS = sys.argv[4] #1
#NUM_DECODER_LAYERS = sys.argv[4] #1
# seed is 6
# bsz 5

dim=200
layers=2
bsz=1

for i in {0..4}; do
    log_dir=../logs_alttransformer/experiment_2/dim${dim}_layers${layers}_bsz${bsz}
    mkdir -p $log_dir
    log_file=$log_dir/log_${i}.txt
    echo "Starting run ${i} for transformer model"
    python ../../src/scan/alttransformer.py ${train_file} ${valid_file} ${dim} ${layers} ${bsz} ${i} > $log_file
    tail -n 1 $log_file
    echo "--"
done




