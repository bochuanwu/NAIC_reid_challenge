export CUDA_VISIBLE_DEVICES=3
python test.py validate --pretrained_choice self \
               --model_name MGN \
               --attention False \
               --SIZE_TEST [384,128] \
               --PIXEL_MEAN [0.0973,0.1831,0.2127] --PIXEL_STD [0.0860,0.0684,0.0964] \
               --norm True \
               --NUM_CLASS 3508 \
               --re_ranking False \
               --eval_flip True \
               --crop_validation True \
               --test_batch 96 \
               --feat 512 \
               --neck_feat after \
               --pretrained_model /data/zhoumi/REID/tx_challenge/pytorch-ckpt/mgn_ibn_bnneck_eraParam_feat512_sepbn_margin1.2/model_best.pth.tar
