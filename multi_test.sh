export CUDA_VISIBLE_DEVICES=7
python test.py multi_test --pretrained_choice self \
               --model_name  resnet50_ibn_a\
               --NUM_CLASS 2465 \
               --norm True \
               --eval_flip True \
               --re_ranking True \
               --neck_feat after \
               --pretrained_model /data/zhoumi/train_project/REID/tx_challenge/pytorch-ckpt/r50_ibn_a_bigsize/model_best.pth.tar
