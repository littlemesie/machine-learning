# -*- coding:utf-8 -*-

from numpy import *

class Kmeans(object):

    def __init__(self):
        pass

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
        cluster_assment = mat(zeros((m,2)))
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
                if cluster_assment[i, 0] != min_index:
                    cluster_changed = True
                    cluster_assment[i, :] = min_index, min_dist**2
            # 更新质心的位置
            for cent in range(k):
                pts_in_clust = data[nonzero(cluster_assment[:, 0].A == cent)[0]]
                centroids[cent, :] = mean(pts_in_clust, axis=0)
        return centroids, cluster_assment

    # 二分K均值聚类算法
    def biKmeans(self, data, k):
        m = shape(data)[0]
        cluster_assment = mat(zeros((m,2)))
        centroid0 = mean(data, axis=0).tolist()[0]
        cent_list =[centroid0]
        for j in range(m):
            cluster_assment[j,1] = distMeas(mat(centroid0), data[j,:])**2
        while len(cent_list) < k:
            lowestSSE = inf
            for i in range(len(centList)):
                ptsInCurrCluster = data[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
                centroidMat, splitClustAss = data(ptsInCurrCluster, 2, distMeas)
                sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
                sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
                print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
                if (sseSplit + sseNotSplit) < lowestSSE:
                    bestCentToSplit = i
                    bestNewCents = centroidMat
                    bestClustAss = splitClustAss.copy()
                    lowestSSE = sseSplit + sseNotSplit
            bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
            bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
            print('the bestCentToSplit is: ',bestCentToSplit)
            print('the len of bestClustAss is: ', len(bestClustAss))
            centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids
            centList.append(bestNewCents[1,:].tolist()[0])
            clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
        return mat(centList), clusterAssment
