accelerate launch train_control_lora_flux.py `
  --pretrained_model_name_or_path "black-forest-labs/FLUX.1-dev" `
  --jsonl_for_train "train.jsonl" `
  --output_dir "output\lora-ah2025" `
  --train_batch_size 1 `
  --num_train_epochs 1 `
  --mixed_precision fp16 `
  --resolution 512 `
  --learning_rate 1e-4 `
  --checkpointing_steps 500 `
  --validation_prompt "courgettebroodjes in style #Ah2025" `
  --validation_image "work\converted\old\courgettebroodjes.jpg"



