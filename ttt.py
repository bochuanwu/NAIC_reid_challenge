# import json
#
# with open('./result/submission_example_A.json', 'r') as fd:
#     clusters = json.load(fd)
#     print(len(clusters))
#
#     count =0
#
#     for key in clusters:
#         list = clusters[key]
#         # print(list)
#         # print(len(list))
#         # print(len(set(list)))
#
#         if len(list) != len(set(list)):
#             count +=1
#             print(key)
#
#     print(count)



import torch



model = torch.load('./resnet101_ibn_a.pth.tar')
new_model = {}
for k, v in model.items():
    name = k[7:] # remove `module.`
    new_model[name] = v


torch.save(new_model, './r101_ibn_a.pth')


