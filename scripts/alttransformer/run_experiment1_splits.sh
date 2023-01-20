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

for split in p1 p2 p4 p8 p16 p32 p64; do
    for i in {0..2}; do
    train_file=../../data/SCAN/simple_split/size_variations/tasks_train_simple_${split}.txt
    valid_file=../../data/SCAN/simple_split/size_variations/tasks_test_simple_${split}.txt
     
    log_dir=../logs_alttransformer/experiment_1_splits/dim${dim}_layers${layers}_bsz${bsz}
    mkdir -p $log_dir
    log_file=$log_dir/log_${split}_${i}.txt
    echo "Starting run ${i} for transformer model on split ${split}"
    python ../../src/scan/alttransformer.py ${train_file} ${valid_file} ${dim} ${layers} ${bsz} ${i} ${lr} > $log_file
    tail -n 1 $log_file
    echo "--"
    done
done



