# lfm
参考[基于矩阵分解的推荐算法](https://www.cnblogs.com/zhangchaoyang/articles/5517186.html)
## 关于超参数
- 学习率一般从0.1开始往上和往下进行试探，样本量不同，最优学习率也不同
- 试验中学习率是否随着iteration的增加而衰减，对效果影响不大
- 正则系数从0.01开始调