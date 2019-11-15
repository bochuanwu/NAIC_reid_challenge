import cv2
import shutil
import os
import glob

base_path = '/Users/zhoumi/git-project/dataset/tx_dataset_reid/train/train_set'
train_list = '/Users/zhoumi/git-project/dataset/tx_dataset_reid/train/train_list.txt'
train_fd = open(train_list, 'r')

stastics = []
for i in range(4768):
    stastics.append(0)
lines = train_fd.readlines()
for line in lines:
    image_name = line.split(' ')[0].split('/')[-1]
    pid = eval(line.split(' ')[-1].replace('\n', ''))
    stastics[pid] +=1
#
#
# for line in lines:
#     image_name = line.split(' ')[0].split('/')[-1]
#     pid = eval(line.split(' ')[-1].replace('\n', ''))
#     if stastics[pid] == 1:
#         src_path = os.path.join(base_path, image_name)
#         shutil.copy(src_path, src_path.replace('train_set', 'train'))

#flip augment
# images = glob.glob(os.path.join('/Users/zhoumi/git-project/dataset/tx_dataset_reid/train/train', '*png'))
#
# for image_path in images:
#     img = cv2.imread(image_path)
#     h_flip = cv2.flip(img, 1)
#     cv2.imwrite(image_path.replace('.png', '_flip.png'), h_flip)

#add augment image in list
train_list = '/Users/zhoumi/git-project/dataset/tx_dataset_reid/train_list.txt'
new_fd = open(train_list, 'w')
pairs = {}
for line in lines:
    new_fd.write(line)
    image_name = line.split(' ')[0].split('/')[-1]
    pid = eval(line.split(' ')[-1].replace('\n', ''))
    if stastics[pid] == 1:
        pairs[image_name] = str(pid)

for key in pairs:
    image_name = 'train/' + key.replace('.png', '_flip.png')
    print(image_name, pairs[key])
    new_fd.write(image_name)
    new_fd.write(' ')
    new_fd.write(pairs[key])
    new_fd.write('\n')





