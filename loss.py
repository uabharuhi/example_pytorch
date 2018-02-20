from torch.autograd import Variable
import torch
loss = torch.nn.CrossEntropyLoss()
print(loss(Variable(torch.Tensor([[0.25,0.25,0.5,0.5],[0.75,0.25,0,0]])),Variable(torch.LongTensor([3,0]))))
#必須要使用Variable ..如果使用Tensor的話因為Tensor並不具備有gradient的功能所以會有error(loss畢竟是要來算gradient的)
#AttributeError: 'torch.FloatTensor' object has no attribute 'requires_grad'