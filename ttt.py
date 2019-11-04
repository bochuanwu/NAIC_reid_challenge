import json

with open('./result/submission_example_A.json', 'r') as fd:
    clusters = json.load(fd)
    print(len(clusters))

    count =0

    for key in clusters:
        list = clusters[key]
        # print(list)
        # print(len(list))
        # print(len(set(list)))

        if len(list) != len(set(list)):
            count +=1
            print(key)

    print(count)


