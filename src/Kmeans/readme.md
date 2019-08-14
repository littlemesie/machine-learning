# Kmeans 聚类算法
- 优点: 容易实现
- 缺点: 可能收敛到局部最小值,在大规模数据集上收敛较慢
- 适用数据类型: 数值型数据
  **k-means是发现给定数据集的K个簇的算法.簇个数K是用户给定的,每一个簇通过其'质心(centroid)',即簇中所有点的中心来描述**
### K-means的工作流程
- 首先,随机确定K个初始点作为质心.然后将数据集中的每个点分配到一个簇中,具体来讲,为每个点找距离最近的质心,并将其分配给该质心所对应的簇.这一步完成之后,每个簇的质心更新为该簇所有点的平均值

- 上述过程伪代码如下

        创建k个点作为起始质心(经常是随机选择)
        当任意一个点的簇分配结果发生改变时
        对数据集中的每个数据点
            对每个质心
                计算质心与数据点之间的距离
            将数据点分配到距其最近的簇
        对每一个簇,计算簇中所有点的均值并将均值作为质心
### k-means的一般流程
1. 收集数据: 使用任意方法
2. 准备数据: 需要数值型数据来计算距离,也可以将标称型数据映射为二值性数据再用于距离计算
3. 分析数据: 使用任意方法
4. 训练算法: 不适用与无监督学习,即无监督学习没有训练过程
5. 测试算法: 应用聚类算法,观察结果.可以使用量化的误差指标如误差平方和来评价算法的结果
6. 使用算法: 可用用于所希望的任何应用.通常情况下,簇质心可以代表整个簇的数据来做出决策
### 使用后处理来提高聚类性能
- 在包含簇分配结果的矩阵中保存着每个点的误差,即该点到簇质心的距离平方值.这个误差可以确定用户预先定义的参数K是否正确,也可以确定生成的簇是否较好
- SSE(Sum of Squared Error,误差平方和):一种用于度量聚类效果的指标.
- SSE值越小表示数据点越接近于它们的质心,聚类效果也越好.因为对误差取了平方,因此更重视那些远离中心的点.一种肯定可以降低SSE值的方法事增加簇的个数,但这违背了聚类的目标.聚类的目标事在保持簇数据不变的情况下提高簇的质量
- 为了保持簇总数不变,可以将两个簇进行合并.可以很容易对二维数据上的聚类进行可视化,如果是多维 的,有两种可以量化的办法:合并最近的质心,或者合并两个使得SSE增幅最小的质心.第一种思路通过计算所有质心之间的距离,然后合并距离最近的两个点来实现.第二种方法需要合并两个簇然后计算总SSE值.必须在所有可能的两个簇上重复上述处理过程,直到找到合并最佳的两个簇为止
### 二分k-means算法
- 二分k-means算法是为了客服k-means算法收敛于局部最小值的问题,二分k-kmeans算法首先将所有点作为一个簇,然后将该簇一分为二.之后选择其中一个簇继续进行划分,选择哪一个簇进行划分取决于对其划分是否可以最大程度降低SSE值.上述基于SSE的划分过程不断重复,直到得到用户指定的簇数目为止

- 二分k-means的伪代码形式如下:

        将所有点看成一个簇
        当簇数目小于k时
          对每一个簇
            计算总误差
            在给定的簇上面进行k-means(k=2)
            计算将该簇一分为二之后的总误差
          选择使得误差最小的那个簇进行划分操作    

### 相关公式
**欧式距离公式:** $d=\sqrt{(xA_0-xB_0)^2+(xA_1-xB_1)^2