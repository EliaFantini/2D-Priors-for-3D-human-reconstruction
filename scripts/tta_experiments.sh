echo "Running TTA experiments with Vanilla baseline"
python -m apps.train_tta --dataroot /scratch/izar/ckli/rendered_jiff_complete   --load_netG_checkpoint_path /scratch/izar/ckli/PIFu/baseline_G/vanilla-baseline/netG_epoch_5  --batch_size 16 --tta_adapt_sample 5 --num_epoch 5 --name tta_baseline_5_sample
python -m apps.train_tta --dataroot /scratch/izar/ckli/rendered_jiff_complete   --load_netG_checkpoint_path /scratch/izar/ckli/PIFu/baseline_G/vanilla-baseline/netG_epoch_5  --batch_size 16 --tta_adapt_sample 10 --num_epoch 5 --name tta_baseline_10_sample
python -m apps.train_tta --dataroot /scratch/izar/ckli/rendered_jiff_complete   --load_netG_checkpoint_path /scratch/izar/ckli/PIFu/baseline_G/vanilla-baseline/netG_epoch_5  --batch_size 16 --tta_adapt_sample 15 --num_epoch 5 --name tta_baseline_15_sample


echo "Running TTA experiments with clip addition baseline"

python -m apps.train_tta --dataroot /scratch/izar/ckli/rendered_jiff_complete --batch_size 16  --num_epoch 5 --name tta_clip_baseline_5_samples --feature_fusion add --learning_rate 0.001 --name tta_baseline_15_sample --tta_adapt_sample 5 --load_netG_checkpoint_path /scratch/izar/ckli/PIFu/baseline_clip_G/clip_baseline/netG_epoch_5
python -m apps.train_tta --dataroot /scratch/izar/ckli/rendered_jiff_complete --batch_size 16  --num_epoch 5 --name tta_clip_baseline_10_samples --feature_fusion add --learning_rate 0.001 --name tta_baseline_15_sample --tta_adapt_sample 10 --load_netG_checkpoint_path /scratch/izar/ckli/PIFu/baseline_clip_G/clip_baseline/netG_epoch_5
python -m apps.train_tta --dataroot /scratch/izar/ckli/rendered_jiff_complete --batch_size 16  --num_epoch 5 --name tta_clip_baseline_15_samples --feature_fusion add --learning_rate 0.001 --name tta_baseline_15_sample --tta_adapt_sample 15 --load_netG_checkpoint_path /scratch/izar/ckli/PIFu/baseline_clip_G/clip_baseline/netG_epoch_5