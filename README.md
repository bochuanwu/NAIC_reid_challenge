# tx_reid_challenge
首届“全国人工智能大赛”（行人重识别 Person ReID 赛项）

1、配置参数
config.py中包含所有的可配置参数：包括backbone、loss、优化参数等

2、模型训练
执行train.sh即可以开始训练
sh train.sh

3、模型推理
执行validate.sh可以进行测试集rank1、map定量化评估，以及进行re-ranking参数进行自动化选择
sh validate.sh

执行test.sh，可直接生成大赛json文件
sh test.sh
