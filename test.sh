export CUDA_VISIBLE_DEVICES=1
python test.py test --pretrained_choice self \
               --model_name resnet50_ibn_a \
               --pretrained_model /data/zhoumi/train_project/REID/tx_challenge/pytorch-ckpt/ibn_margin/checkpoint_ep150.pth.tar
