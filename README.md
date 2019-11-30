# 一、项目所需环境与运行方式

## 项目的文件结构
```
+-- tx_challenge
|   +-- train_set/
|   +-- query_a/
|   +-- gallery_a/
|   +-- query_b/
|   +-- gallery_b/
|   +-- train_list_new.txt
|   +-- val_gallery_list.txt
|   +-- val_query_list.txt
```

## 项目的运行步骤
- `git clone https://github.com/maliho0803/NAIC_reid_challenge.git`
- `cd NAIC_reid_challenge`

### 单模模型训练与推理

1. 执行train.sh即可以开始训练
	- `sh train.sh`
2. 执行validate.sh可以进行测试集rank1、map定量化评估，以及进行re-ranking参数进行自动化选择
	- `sh validate.sh`
3. 执行test.sh，可直接生成大赛json文件
	- `sh test.sh`

### 单模型运行结果的位置
- `./result/submission_example_A.json`


### 多模型权重与对应测试集的距离矩阵信息

- `mkdir model_results_B`

- 模型已上传到百度云网盘 链接: https://pan.baidu.com/s/1tvMdlbaaH_jT6zBbwe-5_w 提取码: etw4 
- 下载好的距离矩阵信息移动到model_results_B下

### 多模型融合提交结果的复现(多模型融合 生成提交结果)
- `python ensemble.py`

### 运行结果的位置
- 项目的根路径下生成 final_submit.json

# 二、数据和模型使用
## 预训练模型的使用情况

- 仿MGN形式或者MGN魔改版的采用resnet50-ibn-a作为预训练的模型
- DDA采用的是se-resnet101-ibn-a作为预训练的模型

## 相关论文及预训练模型的下载链接

- 相关论文 ECCV2018：http://openaccess.thecvf.com/content_ECCV_2018/papers/Xingang_Pan_Two_at_Once_ECCV_2018_paper.pdf 
- 论文作者提供的预训练模型： https://drive.google.com/drive/folders/1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S

# 三、项目运行环境
## 项目所需的工具包/框架
- numpy: 1.16.4
- tensorboardX: 1.9
- pytorch: 1.2.0
- tensorflow: 1.14.0
- torchvision: 0.4.0
- PIL: 6.1.0

## 项目运行的资源环境
- 4卡11G 2080Ti
