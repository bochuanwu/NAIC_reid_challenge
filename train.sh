export CUDA_VISIBLE_DEVICES=1
python train.py train --label_smooth on \
                      --loss_type triplet \
                      --train_batch 128 \
                      --margin None \
                      --model_name resnet50_ibn_a \
                      --last_stride 1\
                      --bnneck bnneck \
                      --pretrained_model /home/zhoumi/.torch/models/r50_ibn_a.pth \
                      --max_epoch 150
