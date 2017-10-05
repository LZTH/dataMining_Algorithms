# -*- coding: UTF-8 -*-  
import numpy  
import random  
import codecs  
import copy  
import re  
import matplotlib.pyplot as plt

class KMeans():

    def __init__(self,infile,k):
        self.data_set=self.Load_Data_Set(infile)        #从数据集获得点集data_set = [[x1,y1,z1,...], [x2,y2,z2,...], [x3,y3,z3,...], ...]
        self.centroid_list=list()                       #存放聚类中心的列表centroid_list = [[x1,y1,z1,...], [x2,y2,z2,...], [x3,y3,z3,...], ...]
        self.clusters_dict=dict()                       #存放聚类结果，结构clusterdirt{cluster1:[[x1,y1,z1,...],[x2,y2,z2,...],...],cluster2:[[x1,y1,z1,...],[x2,y2,z2,...]],...}
        self.var=0.0                                    #记录聚类的簇内变差

    def Calcular_Distance(self,vec1, vec2):  
        ''' 计算向量vec1和向量vec2之间的欧氏距离 ''' 
        return numpy.sqrt(numpy.sum(numpy.square(vec1 - vec2)))  
    
    def Load_Data_Set(self,infile):  
        '''载入数据测试数据集inFile'''
        #plt.figure("points")            #定义作图的标题
        indate = codecs.open(infile, 'r', 'utf-8').readlines()     
        data_set = list()      #数据集
        for line in indate:  
            line = line.strip()     #删除空白符（包括'\n', '\r',  '\t',  ' ')
            str_list = re.split('[ ]+', line)  # 去除多余的空格  
            num_list = list()     #一个坐标点
            for item in str_list:  
                num = float(item)      
                num_list.append(num)  
            data_set.append(num_list)
            #plt.plot(num_list[0],num_list[1],'ok')    #将点加画到坐标轴上
        #plt.show()               #将点可视化
        return data_set      # data_set = [[x1,y1,z1,...], [x2,y2,z2,...], [x3,y3,z3,...], ...]
    
    def Init_Centroids(self,k):  
        ''' 初始化k个质心，随机获取 ''' 
        return random.sample(self.data_set, k)  # 从data_set中随机获取k个数据项返回  
    
    def Get_Clusters(self):  
        ''' 对每个属于data_set的item，计算item与centroidList中k个质心的欧式距离，找出距离最小的，  
        并将item加入相应的簇类中 ''' 
    
        clusters={}                          #存放聚类结果           
        for item in self.data_set:  
            vec1 = numpy.array(item)         # 转换成array形式  
            flag = 0                         # 簇分类标记，记录与相应簇距离最近的那个簇  
            min_dis = float("inf")           # 到所有簇心的最小距离，初始化为最大值  
    
            for i in range(len(self.centroid_list)):  
                vec2 = numpy.array(self.centroid_list[i])  
                distance = self.Calcular_Distance(vec1, vec2)    # 计算相应的欧式距离  
                if distance < min_dis:      
                    min_dis = distance  
                    flag = i                                # 循环结束时，flag保存的是与当前item距离最近的那个簇标记  
    
            if flag not in clusters.keys():                 # 簇标记不存在，进行初始化  
                clusters[flag] = list()  
            clusters[flag].append(item)                     # 加入相应的类别中  
    
        return clusters                                     # 返回新的聚类结果  
    
    def Get_Centroids(self):  
        ''' 从聚类结果得到k个质心'''  
        centroid_list_temp = list()  
        for key in self.clusters_dict.keys():  
            centroid = numpy.mean(numpy.array(self.clusters_dict[key]), axis = 0)  # 计算每列的均值，即找到质心  
            # print key, centroid  
            centroid_list_temp.append(centroid)  
        
        return numpy.array(centroid_list_temp).tolist()  
    
    def Get_Var(self):  
        ''' 计算簇集合间的均方误差 将簇类中各个向量与质心的距离进行累加求和'''  
    
        sum = 0.0  
        for key in self.clusters_dict.keys():  
            vec1 = numpy.array(self.centroid_list[key])  
            distance = 0.0  
            for item in self.clusters_dict[key]:  
                vec2 = numpy.array(item)  
                distance += self.Calcular_Distance(vec1, vec2)  
            sum += distance  
        return sum  
    
    def Show_Clusters(self):  
        ''' 展示聚类结果 '''  
        plt.figure("clusters")
        colorMark = ['or', 'ob', 'og', 'ok', 'oy', 'ow']      # 不同簇类的标记 'or' --> 'o'代表圆，'r'代表red，'b':blue  
        centroidMark = ['dr', 'db', 'dg', 'dk', 'dy', 'dw']   # 质心标记 同上'd'代表棱形  
        for key in self.clusters_dict.keys():  
            plt.plot(self.centroid_list[key][0], self.centroid_list[key][1], centroidMark[key], markersize = 12)  # 画质心点  
            for item in self.clusters_dict[key]:  
                plt.plot(item[0], item[1], colorMark[key]) # 画簇类下的点  
    
        plt.show()

    def Main(self,infile,k):
        '''迭代进行聚类，获得最终聚类结果'''
        self.centroid_list = self.Init_Centroids(k)          # 初始化质心，设置k=4  
        self.clusters_dict = self.Get_Clusters()  # 第一次聚类迭代  
        new_var = self.Get_Var()          # 获得均方误差值，通过新旧均方误差来获得迭代终止条件  
        old_var = -0.0001                                    # 旧均方误差值初始化为-1    
        step = 2  
        while abs(new_var - old_var) >= 0.0001:              # 当连续两次聚类结果小于0.0001时，迭代结束            
            self.centroid_list = self.Get_Centroids()          # 获得新的质心  
            self.clusters_dict = self.Get_Clusters()  # 新的聚类结果  
            old_var = new_var                                     
            new_var = self.Get_Var()    
            step += 1
        self.var=new_var

    def step_show(self,infile,k):
        '''逐步进行聚类并将每一步可视化'''
        self.centroid_list = self.Init_Centroids(k)          # 初始化质心，设置k=4  
        self.clusters_dict = self.Get_Clusters()  # 第一次聚类迭代  
        new_var = self.Get_Var()          # 获得均方误差值，通过新旧均方误差来获得迭代终止条件  
        old_var = -0.0001                                    # 旧均方误差值初始化为-1  
        print('***** 第1次迭代 *****\n')   
        print('簇类')  
        for key in self.clusters_dict.keys():  
            print (key, ' --> ', self.clusters_dict[key])  
        print('k个簇心坐标: ', self.centroid_list)  
        print('平均均方误差: ', new_var,"\n")  
        self.Show_Clusters()             # 展示聚类结果  

        step = 2  
        while abs(new_var - old_var) >= 0.0001:              # 当连续两次聚类结果小于0.0001时，迭代结束            
            self.centroid_list = self.Get_Centroids()          # 获得新的质心  
            self.clusters_dict = self.Get_Clusters()  # 新的聚类结果  
            old_var = new_var                                     
            new_var = self.Get_Var()  

            print('***** 第%d次迭代 *****\n' % step)  
            print ('簇类')  
            for key in self.clusters_dict.keys():  
                print( key, ' --> ', self.clusters_dict[key])  
            print ('k个簇心坐标: ', self.centroid_list)  
            print ('平均均方误差: ', new_var,'\n') 
            self.Show_Clusters()            # 展示聚类结果  
            step += 1
        self.var=new_var

if __name__=="__main__":
    cluster=KMeans('infile.txt',4)
    cluster.Main('infile.txt',4)
    print(cluster.clusters_dict,'\n',cluster.centroid_list,'\n',cluster.var)



