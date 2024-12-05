## Pytorch BiLSTM_CRF医疗命名实体识别
运行程序之前要注意学习一些基础知识：如什么是命名实体识别、BIO标注、BERT模型、BiLSTM模型、CRF模型

这个项目是阿里天池大赛的一个赛题，《瑞金医院MMC人工智能辅助构建知识图谱大赛》，所以数据集是公开的
要求选手在糖尿病相关的学术论文和临床指南的基础上，做实体的标注，也就是NLP领域的命名实体识别任务。

天池赛题地址：https://tianchi.aliyun.com/competition/entrance/231687/information

### 目录结构

input
    存放数据集，即用于训练和预测的医学文本。
    origin为原始文件，上百个文档，其中xx.txt为文本文档(原始文本),xx.ann存放标记的命名实体（仅正样本的各类信息，包括位置信息、实体类型等）
    
output
    存放输出的文件
    包括，
        annotation 是原始输入文件.txt经过标注后的文本文档
        tran_sample.txt 拆分后的训练集 供train.py训练
        test_sample.txt 拆分后的测试集test.py会用到得到模型识别的P、R、F1值等
        label.txt 命名实体识别所定义的实体类型字典
        vocab.txt 字符词典及对应序号
各类脚本
    xx.py

#各类脚本执行顺序
config.py
    配置文件：写一些文件路径、训练参数等
    
data_process.py
    数据预处理：生成一些output中用到的文件
        
utils.py
    工具:加载词表、初始预训练模型BERT等
    
models.py
    结合BERT、BiLSTM、CRF
    
train.py
    训练脚本，由于我的电脑GPU不大够用，所以用Kaggle训练的，具体可搜索kaggle训练进行学习，当然训练的时候需要手动修改train.py。
    因为我记得这个项目一般训练50轮次即可达到与100轮次差不多的效果，kaggle输出控件是有限的。
    但是每次训练会生成一个1.25G的文件太占输出，所以我进行了修改，采用断点训练的方式，训练过程还保存在kaggle网站上。
    
test.py
    训练之后获得的模型文件应用于测试脚本，用于获取精确率、召回率、F1值等
    
predict.py
    使用训练之后的模型文件进行预测，可输入一段文字，返回这段文字中存在的命名实体及标签
    
现在这个项目还缺两个东西
1.BERT模型，HUggingFace的BERT文件大概有几百兆，为了方便文件传输，后续再新增
2.我已经训练好的BERT+BiLSTM+CRF模型，这个模型有1.25G，因为前段时间电脑磁盘都满了，所以删除掉了一些，需要的话我得从百度网盘下载一下
        
    

