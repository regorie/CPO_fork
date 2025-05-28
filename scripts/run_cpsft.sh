export PATH=$PATH:/home/jeeves/.local/bin

pip install datasets -U
pip install deepspeed
pip install peft
pip install -U transformers

deepspeed --num_gpus=2 ./src/CPSFT/cpsft/train_sft.py\
 --data_path ./Half_sft.json \
 --deepspeed ./src/CPSFT/cpsft/deepspeed_config/ZeRO_3.json \
 --output_dir ./data/checkpoints/mistral_sft/ \
 --eval_steps 20 \
 --save_steps 100 \
 --base_model mistralai/Mistral-7B-Instruct-v0.2 \
 --prompt_template_name mistral_delete




