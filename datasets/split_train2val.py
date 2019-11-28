import glob
import random

# val_num = 500
# train_list = '/Users/zhoumi/git-project/dataset/tx_dataset_reid/train/train_list.txt'
# train_list1 = '/Users/zhoumi/git-project/dataset/tx_dataset_reid/train_list_new1.txt'
# val_query_list = '/Users/zhoumi/git-project/dataset/tx_dataset_reid/val_query_list.txt'
# val_gallery_list = '/Users/zhoumi/git-project/dataset/tx_dataset_reid/val_gallery_list.txt'

# train_fd = open(train_list, 'r')
# train_fd1 = open(train_list1, 'r')
# print(len(train_fd.readlines()), len(train_fd1.readlines()))

# val_q_fd = open(val_query_list, 'r')
# val_g_fd = open(val_gallery_list, 'r')

# train = len(train_fd.readlines())
# gallery = len(val_g_fd.readlines())
# query = len(val_q_fd.readlines())
#
# print(train, gallery, query, train+gallery+query)

# lines = train_fd.readlines()
# last_pid = 0
# count = 0
# for line in lines:
#     pid = eval(line.split(' ')[-1].replace('\n', ''))
#     image_name = line.split(' ')[0].split('/')[-1]
#     if pid != last_pid:
#         count +=1
#
#     last_pid = pid
#     train_fd1.write(image_name)
#     train_fd1.write(' ')
#     train_fd1.write(str(count))
#     train_fd1.write('\n')
#
# train_fd.close()
# train_fd1.close()




# with open('/Users/zhoumi/git-project/dataset/tx_dataset_reid/train/train_list_new.txt', 'r') as fd:
#
#     # train_fd = open(train_list, 'w')
#     # val_q_fd = open(val_query_list, 'w')
#     # val_g_fd = open(val_gallery_list, 'w')
#
#     lines = fd.readlines()
#     last_pid = 0
#     t_w = True
#     for line in lines:
#         pid = eval(line.split(' ')[-1].replace('\n', ''))
#         image_name = line.split(' ')[0].split('/')[-1]
#
#         if pid == last_pid:
#             if t_w:
#                 train_fd.write(image_name)
#                 train_fd.write(' ')
#                 train_fd.write(str(pid))
#                 train_fd.write('\n')
#             else:
#                 val_g_fd.write(image_name)
#                 val_g_fd.write(' ')
#                 val_g_fd.write(str(pid))
#                 val_g_fd.write('\n')
#
#         else:
#             rad = random.random()
#             if rad < 0.1:
#                 t_w = False
#                 val_q_fd.write(image_name)
#                 val_q_fd.write(' ')
#                 val_q_fd.write(str(pid))
#                 val_q_fd.write('\n')
#
#             else:
#                 t_w = True
#                 train_fd.write(image_name)
#                 train_fd.write(' ')
#                 train_fd.write(str(pid))
#                 train_fd.write('\n')
#
#         last_pid = pid

    # import collections
    # stastic = []
    # for i in range(4768):
    #     stastic.append(0)
    # lines = fd.readlines()
    # print(len(lines))
    # for line in lines:
    #     pid = eval(line.split(' ')[-1].replace('\n', ''))
    #     stastic[pid - 1] +=1
    #
    # ids = []
    # for i in range(len(stastic)):
    #     if stastic[i] == 1:
    #         ids.append(i+1)
    #
    # print(ids)
    #
    # for line in lines:
    #     pid = eval(line.split(' ')[-1].replace('\n', ''))
    #     image_name = line.split(' ')[0].split('/')[-1]
    #
    #     if pid in ids:
    #         val_g_fd.write(image_name)
    #         val_g_fd.write(' ')
    #         val_g_fd.write(str(pid))
    #         val_g_fd.write('\n')
    #
    #     else:
    #         train_fd.write(image_name)
    #         train_fd.write(' ')
    #         train_fd.write(str(pid))
    #         train_fd.write('\n')


# add test into train

# lines = train_fd.readlines()
#
# for line in lines:
#     train_fd1.write(line)
#
#
# query_path = '/Users/zhoumi/git-project/dataset/tx_dataset_reid/test/query_a_list.txt'
# count = 2465
# print(count)
# with open(query_path, 'r') as fd:
#     query_lines = fd.readlines()
#     for query_line in query_lines:
#         image_name = query_line.split(' ')[0].split('/')[-1]
#         train_fd1.write(image_name)
#         train_fd1.write(' ')
#         train_fd1.write(str(count))
#         train_fd1.write('\n')
#         count +=1

# train data stastic

# stastics = []
# for i in range(4768):
#     stastics.append(0)
# lines = train_fd.readlines()
# for line in lines:
#     image_name = line.split(' ')[0]
#     pid = eval(line.split(' ')[-1].replace('\n', ''))
#     stastics[pid] +=1
#
# print(stastics)
# print(stastics.index(max(stastics)), max(stastics), min(stastics))


fd = open('/Users/zhoumi/git-project/dataset/tx_dataset_reid/train_list_new_test.txt', 'a')

img_paths = glob.glob('/Users/zhoumi/git-project/dataset/tx_dataset_reid/pesudo/*png')
print(len(img_paths))

pairs = {}
rid = 2464
for img_path in img_paths:
    img_name = img_path.split('/')[-1]
    id = img_path.split('/')[-1].split('_')[0]
    print(img_name, id)
    if id in pairs:
        rid = pairs[id]
    else:
        rid +=1
        pairs[id] = rid
    # fd.write(img_name)
    # fd.write(' ')
    # fd.write(str(rid))
    # fd.write('\n')




