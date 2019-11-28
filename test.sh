export CUDA_VISIBLE_DEVICES=2
python test.py test --pretrained_choice self \
               --model_name MGN \
               --attention False \
               --NUM_CLASS 3508 \
               --SIZE_TEST [384,128] \
               --PIXEL_MEAN [0.0973,0.1831,0.2127] --PIXEL_STD [0.0860,0.0684,0.0964] \
               --norm True \
               --eval_flip True \
               --re_ranking True \
               --crop_validation True \
               --neck_feat after \
               --feat 512 \
               --pretrained_model /data/zhoumi/REID/tx_challenge/pytorch-ckpt/mgn_ibn_bnneck_eraParam_feat512_sepbn_margin1.2/model_best.pth.tar
