# lfm
参考[基于矩阵分解的推荐算法](https://www.cnblogs.com/zhangchaoyang/articles/5517186.html)
## 关于超参数
- 学习率一般从0.01开始往上和往下进行试探，样本量不同，最优学习率也不同
- 试验中学习率是否随着iteration的增加而衰减，对效果影响不大
- 如果出现NaN，很可能是因为你的学习率设大了，调小一些
- 正则系数从0.01开始调
## 关于并行训练
模型参数是用slice存储的，相比于map它有2个优势：
1. 节约内存
2. go里面slice并发读写不会panic，但map会。当然不加锁地并发读写slice会存在脏读脏写的问题，但这点数据不一致对模型收敛不会造成影响（[看图](https://github.com/Orisun/lfm/blob/master/train_err.png)），分布式训练框架也都是这么干的。