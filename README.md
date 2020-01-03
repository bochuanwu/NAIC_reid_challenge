# NAIC2019-Person ReID

首届“全国人工智能大赛”（行人重识别 Person ReID 赛项）

# environment
Make sure your conda is installed.

- conda create --name reid python=3.6
- source activate reid
- conda install pytorch torchvision cudatoolkit -c pytorch
- pip install -r requirements.txt
- (optional) 
```
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext 
```

# data
- Down train set and test test from NAIC2019
- mkdir input
- ln -s ~/复赛训练集 ./input/second_stage_train
- ln -s ~/复赛测试集 ./input/test 
- python data_split.py

```
+-- input/
|   +-- test/
|       +-- query_a/
|       +-- gallery_a/
|       +-- query_b/
|       +-- gallery_b/
|   +-- train/
|       +-- second_stage_train/
|       +-- second_stage_train_list_refine.txt
|       +-- train.txt
|       +-- query.txt
|       +-- gallery.txt
```

- if you want to add pseudo label for training, 
you can add test images to train_set 
and then add this image names and pseudo label to train.txt

# build metric function
- cd ./metrics/rank_cylib
- python setup.py build_ext --inplace

# train, val or inference mode
- If you want to train, please modify is_train = True
```
data:
  type: 'image'
  sources: ['kescireid']
  targets: ['kescireid']
  is_train: True
```

- If you want to val, please modify data.is_train = True and test.evaluate = True
```
data:
  type: 'image'
  sources: ['kescireid']
  targets: ['kescireid']
  is_train: True
test:
  evaluate: True # train stage
```

- If you want to inference and get sumbit.json, please modify data.is_train = False
```
data:
  type: 'image'
  sources: ['kescireid']
  targets: ['kescireid']
  is_train: False
```

# how to run
- export CUDA_VISIBLE_DEVICES=0, 1
- python main.py --config_file config.yaml

# project tree
```
+-- naic2019-person-reid
|   +-- configs/
|   +-- data/
|   +-- engine/
|   +-- losses/
|   +-- input/
|   +-- metrics/
|   +-- optim
|   +-- utils
|   +-- default_config.py
|   +-- main.py
|   +-- README.md
|   +-- requirements.txt
|   +-- script.sh
```
