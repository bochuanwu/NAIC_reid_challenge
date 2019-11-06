export CUDA_VISIBLE_DEVICES=2
python test.py validate --pretrained_choice self \
               --model_name resnet101_ibn_a \
               --norm False \
               --re_ranking False \
               --eval_flip False \
               --test_batch 96 \
               --pretrained_model /data/zhoumi/train_project/REID/tx_challenge/pytorch-ckpt/r101_ibn_a_instance3/model_best.pth.tar
