import glob
import os

gallery_path = '/data/zhoumi/datasets/reid/tx_challenge/gallery_a'

with open('/data/zhoumi/datasets/reid/tx_challenge/gallery_a_list.txt', 'w') as fd:
    imgfiles = glob.glob(os.path.join(gallery_path, '*png'))

    for imgfile in imgfiles:
        name = 'gallery_a/' + imgfile.split('/')[-1]
        print(name)
        fd.write(name)
        fd.write(' ')
        fd.write(str(0))
        fd.write('\n')