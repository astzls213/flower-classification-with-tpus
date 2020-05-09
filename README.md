# Flowers classification with tpus

## 项目架构

Analyse文件夹：包含对数据集的分析，以及用不同的网络架构对性能的影响

data文件夹：包含网络架构文件，日志文件，以及class_weight参数

deprecate文件夹：包含已经弃用的py文件，多用于测试

hyper_opt文件夹：超参数优化过程及结果

models文件夹：包含每个架构整个流程的代码和模型

performance文件夹，每个架构训练后的结果（acc, val_acc, loss, val_loss, f1 score, confusion matrix)

utilities文件夹：一些对超参数优化有帮助的可视化小脚本

## 开发过程

### （1）自定义网络ZNN

