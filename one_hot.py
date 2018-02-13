import torch
# http://blog.csdn.net/u012436149/article/details/77017832
# https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
#http://pytorch.org/docs/master/tensors.html --> scatter_
#http://blog.csdn.net/VictoriaW/article/details/72874637
#http://blog.csdn.net/u012436149/article/details/77017832

#scatter 將一個tensor在某個維度上面分散給另外一個tensor
#可以選擇不要每一個值都scatter
#test = [2,3,6,14,9]
#max_num  = max(test)
#n = len(test)
#res = torch.zeros(n, max_num)
#res.scatter_(0, torch.LongTensor(test),torch.FloatTensor([1 for x in range(len(res))]) )
#print(res)


# x:    [[0,0,0],
#       [0,0,0],
#       [0,0,0]]
#
# y:    [[87,13],
#       [88,32] ]
# dim : 0
# index :[[1,2],    --> 87到1 ,88到2 第一個col結束
#         [2,0]]    --->13到2 ,32到0 第二個col結束
# 把y scatter到x在dim0上面並且以index指定的方式
# 不一定需要一樣長度
#
# all values in a row along the specified dimension dim must be unique.

#self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
#self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
#self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
#
#
# dim 的那個維度 source 和 index還有 value要是一樣數量
t = torch.zeros(3, 3).scatter_(0, torch.LongTensor([[0,1,2], [2,0,1]]), torch.FloatTensor([[87,13,11], [88,32,-1]]))
print(t)
onehot = torch.zeros(3, 3).scatter_(0, torch.LongTensor([[1,2,0]]), torch.FloatTensor([[1,1,1]]))
print(onehot)
