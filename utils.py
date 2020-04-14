import os
import re
import torch

def save_checkpoint(state, iters, tag='', path = './models'):
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path+"/{}checkpoint-{:06}.pth.tar".format(tag, iters))
    torch.save(state, filename)

def get_lastest_model(path='./models', exact = None):
    if not os.path.exists(path):
        os.mkdir(path)
    model_list = os.listdir(path+'/')
    if exact is not None:
        model_list = [x for x in model_list if exact in x]
    if model_list == []:
        return None, 0
    model_list.sort()
    lastest_model = model_list[-1]
    iters = re.findall(r'\d+', lastest_model)
    return path + '/' + lastest_model, int(iters[0])