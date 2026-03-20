cd /mnt/nas/zhangxuheng/last0/scripts
python visualize_latent_recon.py \
  --checkpoint_path /mnt/data/zhangxuheng/ckpt/exp/libero_spatial_ablation/sim_pixel_recon_ls16/checkpoint-34-115885/tfmr \
  --data_path /mnt/data/zhangxuheng/data/libero_training_data/libero_json/libero_spatial_no_noops_view2_chunk4_16_stride8_fast1_sparse_fastslow_train.json \
  --data_root "" \
  --output_dir /mnt/nas/zhangxuheng/last0/ \
  --num_samples 1 \
  --latent_size 16 \
  --action_dim 7 \
  --action_chunk 8