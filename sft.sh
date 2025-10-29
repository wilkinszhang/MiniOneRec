export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
# Office_Products, Industrial_and_Scientific
for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}
    
    torchrun --nproc_per_node 8 \
            train.py \
            --base_model /home/huangyanwen.hyw/LLM_models/qwen2.5-1.5b-Instruct \
            --batch_size 1024\
            --micro_batch_size 16 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir output_dir/1020_Industrial_sft_1.5b \
            --wandb_project wandb_proj \
            --wandb_run_name wandb_name \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path xxx/Industrial_and_Scientific/Industrial_and_Scientific.index.json \
            --item_meta_path xxx/Industrial_and_Scientific/Industrial_and_Scientific.item.json
done
