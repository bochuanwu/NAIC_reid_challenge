export CUDA_VISIBLE_DEVICES=4
python test.py test --pretrained_choice self \
               --model_name resnet101_ibn_a \
               --norm False \
               --eval_flip False \
               --re_ranking False \
               --pretrained_model /data/zhoumi/train_project/REID/tx_challenge/pytorch-ckpt/r101_ibn_a_instance3/checkpoint_ep150.pth.tar
