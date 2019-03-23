# -*- coding:utf-8 -*-

from numpy import *
import numpy as np

class Kmeans(object):

    def __init__(self):
        pass

    def load_data(self, file_name):
        '''
        加载数据集
        :param file_name:
        :return:
        '''
        trian = []
        with open(file_name,'r') as f:
            for line in f.readlines():
                cur_line = line.strip().split('\t')
                # 将数据转换为浮点类型,便于后面的计算
                flt_line = list(map(float, cur_line))  # 映射所有的元素为 float（浮点数）类型
                trian.append(flt_line)
        return trian

    def dist_eclud(self, vec_a, vec_b):
        """计算欧氏距离"""
        return np.sqrt(sum(np.power(vec_a - vec_b, 2)))

    def rand_cent(self, data, k):
        """
       为给定数据集构建一个包含K个随机质心的集合,
       随机质心必须要在整个数据集的边界之内,这可以通过找到数据集每一维的最小和最大值来完成
       然后生成0到1.0之间的随机数并通过取值范围和最小值,以便确保随机点在数据的边界之内
        """
        n = shape(data)[1]
        centroids = mat(zeros((k, n)))
        for j in range(n):
            min_j = min(data[:, j])
            # 计算每一列的范围值
            range_j = float(max(data[:, j]) - min_j)
            # 计算每一列的质心,并将值赋给centroids
            centroids[:, j] = mat(min_j + range_j * random.rand(k, 1))
        return centroids


    def k_means(self, data, k):
        """
        创建K个质心,然后将每个店分配到最近的质心,再重新计算质心。
        这个过程重复数次,直到数据点的簇分配结果不再改变为止
        """

        m = shape(data)[0]
        # 包含两个列: 一列记录簇索引值, 第二列存储误差(误差是指当前点到簇质心的距离, 后面会使用该误差来评价聚类的效果)
        cluster_assment = mat(zeros((m,2)))
        # 创建质心,随机K个质心
        centroids = self.rand_cent(data, k)
        cluster_changed = True
        while cluster_changed:
            cluster_changed = False
            for i in range(m):
                min_dist = inf
                min_index = -1
                # 寻找最近的质心
                for j in range(k):
                    dist_ij = self.dist_eclud(centroids[j, :], data[i, :])
                    if dist_ij < min_dist:
                        min_dist = dist_ij
                        min_index = j
                # 如果任一点的簇分配结果发生改变,则更新cluster_changed标志
                if cluster_assment[i, 0] != min_index:
                    cluster_changed = True
                    cluster_assment[i, :] = min_index, min_dist**2
            # 更新质心的位置
            for cent in range(k):
                pts_in_clust = data[nonzero(cluster_assment[:, 0].A == cent)[0]]
                centroids[cent, :] = mean(pts_in_clust, axis=0)
        # 返回所有的类质心与点分配结果
        return centroids, cluster_assment

    def bi_kmeans(self, data, k):
        """二分K均值聚类算法"""
        m = shape(data)[0]
        cluster_assment = mat(zeros((m,2)))
        centroid0 = mean(data, axis=0).tolist()
        cent_list =[centroid0]
        for j in range(m):
            cluster_assment[j,1] = self.dist_eclud(mat(centroid0), data[j, :])**2
        while len(cent_list) < k:
            lowest_sse = inf
            for i in range(len(cent_list)):
                curr_cluster = data[nonzero(cluster_assment[:,0].A==i)[0], :]
                centroid_mat, split_clust_ass = data(curr_cluster, 2, self.dist_eclud)
                sse_split = sum(split_clust_ass[:,1])
                sse_not_split = sum(cluster_assment[nonzero(cluster_assment[:,0].A!=i)[0],1])
                print("sse_split, and sse_split: ",sse_split,sse_split)
                if (sse_split + sse_not_split) < lowest_sse:
                    best_cent_to_split = i
                    best_new_cents = centroid_mat
                    best_clust_ass = split_clust_ass.copy()
                    lowest_sse = sse_split + sse_not_split
                    best_clust_ass[nonzero(best_clust_ass[:,0].A == 1)[0],0] = len(cent_list)
            best_clust_ass[nonzero(best_clust_ass[:,0].A == 0)[0],0] = best_cent_to_split
            print('the best_cent_to_split is: ',best_cent_to_split)
            print('the len of best_clust_ass is: ', len(best_clust_ass))
            cent_list[best_cent_to_split] = best_new_cents[0,:].tolist()[0]
            cent_list.append(best_new_cents[1,:].tolist()[0])
            cluster_assment[nonzero(cluster_assment[:,0].A == best_cent_to_split)[0],:]= best_clust_ass
        return mat(cent_list), cluster_assment

if __name__ == '__main__':
    km = Kmeans()
    data = mat(km.load_data('../../data/Kmeans/testSet.txt'))

    p, q = km.k_means(data, 4)
    print(p)