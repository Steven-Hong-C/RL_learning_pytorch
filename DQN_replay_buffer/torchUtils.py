import torch

def one_hot(index_list,class_num):
    if type(index_list) == torch.Tensor:
        index_list = index_list.detach().numpy()
    indexes  = torch.LongTensor(index_list).view(-1,1)
    out = torch.zeros(len(index_list), class_num)
    out = out.scatter_(dim = 1, index = indexes, value = 1)
    return out