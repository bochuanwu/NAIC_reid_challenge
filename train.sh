export CUDA_VISIBLE_DEVICES=1
python train.py train --label_smooth on \
                      --loss_type triplet \
                      --train_batch 96 \
                      --num_instances 3 \
                      --margin None \
                      --model_name resnet101_ibn_a \
                      --last_stride 1\
                      --bnneck bnneck \
                      --pretrained_model /home/zhoumi/.torch/models/r101_ibn_a.pth \
                      --max_epoch 150 \
                      --sampler_new True \
                      --save_dir ./pytorch-ckpt/r101_ibn_a_instance3_new
