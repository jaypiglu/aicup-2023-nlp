export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python run_xnli.py \
  --model_name_or_path hfl/chinese-lert-base \
  --cache_dir /nfs/nas-6.1/wclu/cache \
  --language zh \
  --train_language zh \
  --do_train \
  --do_eval \
  --gradient_accumulation_steps 4 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 64 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --max_seq_length 512 \
  --output_dir ckeckpoints/lert-base \
  --evaluation_strategy epoch \
  --logging_dir runs/lert-base \
  --save_strategy epoch \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --logging_steps 100


