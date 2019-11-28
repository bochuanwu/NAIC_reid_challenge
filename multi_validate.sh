export CUDA_VISIBLE_DEVICES=3
python test.py multi_validate --pretrained_choice self \
               --model_name resnet50_ibn_a \
               --norm True \
               --NUM_CLASS 2465 \
               --re_ranking False \
               --eval_flip True \
               --test_batch 16 \
               --neck_feat after \
               --pretrained_model /data/zhoumi/train_project/REID/tx_challenge/pytorch-ckpt/r50_ibn_a_lr/model_best.pth.tar
