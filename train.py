import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data
import numpy as np
from network import Approx, Dual, Lagrange, objective
from utils import *
import time

class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data
    
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias, 0.01)
        
def weights_init_to_0(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.constant_(m.weight, 0.0)
        
def get_args():
    parser = argparse.ArgumentParser("ShuffleNetV2_Plus")
    parser.add_argument('--batch-size', type=int, default=1024, help='batch size')
    parser.add_argument('--total-iters', type=int, default=int(5e5), help='total iters')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='init learning rate')
    parser.add_argument('--save-path', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('--auto-continue', type=bool, default=True, help='auto continue')
    parser.add_argument('--show-interval', type=int, default=50, help='display interval')
    parser.add_argument('--save-interval', type=int, default=1000, help='save interval')
    parser.add_argument('--power-dB', type=float, default=1, help='power constrain in dB')
    parser.add_argument('--gpu', type=str, default='0,', help='gpu_index')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    save_path = args.save_path
    save_interval = args.save_interval
    show_interval = args.show_interval
    auto_continue = args.auto_continue
    learning_rate = args.learning_rate
    total_iters = int(args.total_iters)
    power = args.power_dB
    
    # t1 = time.time()
    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True

    num_workers = 0

    # DataSetup
    N = 2
    train_batch_size = int(5e3)
    # val_batch_size = int(1e3)
    # train_batch_size = int(5e3)
    val_batch_size = int(1e4)
    
    train_sample_num = train_batch_size*1000
    val_sample_num = val_batch_size*10

    h_train_dataset = torch.zeros(train_sample_num, N)
    h_train_dataset.exponential_(lambd=1)
    h_train_loader = torch.utils.data.DataLoader(
        h_train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_gpu)
    h_train_dataprovider = DataIterator(h_train_loader)

    g_train_dataset = torch.zeros(train_sample_num, N)
    g_train_dataset.exponential_(lambd=1)
    g_train_loader = torch.utils.data.DataLoader(
        g_train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_gpu)
    g_train_dataprovider = DataIterator(g_train_loader)

    h_val_dataset = torch.zeros(val_sample_num, N)
    h_val_dataset.exponential_(lambd=1)
    h_val_loader = torch.utils.data.DataLoader(
        h_val_dataset, batch_size=val_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_gpu)
    h_val_dataprovider = DataIterator(h_val_loader)

    g_val_dataset = torch.zeros(val_sample_num, N)
    g_val_dataset.exponential_(lambd=1)
    g_val_loader = torch.utils.data.DataLoader(
        g_val_dataset, batch_size=val_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_gpu)
    g_val_dataprovider = DataIterator(g_val_loader)

    print('Data Generating Finished!')

    model_Approx = Approx(inp = 2*N, oup = N, hidden_dim = 10*N)
    model_Approx.apply(weights_init)
    optimizer_Approx = torch.optim.Adam(model_Approx.parameters(),
                               lr = learning_rate)

    model_Dual = Dual(inp = N, oup = 1, Gamma = 1, P = 10**(power/10))
    model_Dual.apply(weights_init_to_0)
    optimizer_Dual = torch.optim.Adam(model_Dual.parameters(),
                               lr = learning_rate)
    criterion = Lagrange()

    if use_gpu:
        model_Approx = nn.DataParallel(model_Approx)
        model_Dual = nn.DataParallel(model_Dual)
        loss_function = criterion.cuda()
        device = torch.device("cuda")
    else:
        loss_function = criterion
        device = torch.device("cpu")

    model_Approx = model_Approx.to(device)
    model_Dual = model_Dual.to(device)

    iters = 0
    y1 = np.zeros(total_iters)
    y2 = np.zeros(total_iters)
    x1 = np.zeros(total_iters)
    x2 = np.zeros(total_iters)
    x3 = np.zeros(total_iters)

    if auto_continue:
        lastest_model, lastest_iters = get_lastest_model(path = save_path)
        if lastest_model is not None:
            iters = lastest_iters
            checkpoint = torch.load(lastest_model, map_location=None if use_gpu else 'cpu')
            y1 = checkpoint['val_cost']
            y2 = checkpoint['train_cost']
            x1 = checkpoint['lambda1']
            x2 = checkpoint['lambda2']
            x3 = checkpoint['lambda3']
            model_Approx.load_state_dict(checkpoint['state_approx_dict'], strict=True)
            model_Dual.load_state_dict(checkpoint['state_dual_dict'], strict=True)
            print('load from checkpoint with iters: ', iters)

    # t2 = time.time()
    # print('prepare time: ', t2 - t1)
    while iters < total_iters:

        t3 = time.time()
        iters += 1
        model_Approx.train()
        model_Dual.train()

        h_data = h_train_dataprovider.next()
        g_data = g_train_dataprovider.next()
        h_data, g_data = h_data.to(device), g_data.to(device)

        output_Approx = model_Approx(h_data, g_data)
        output_Dual = model_Dual(output_Approx, g_data)
        loss = loss_function(output_Approx, output_Dual, h_data)
        optimizer_Approx.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_Approx.step()

        output_Approx = model_Approx(h_data, g_data)
        loss = -loss_function(output_Approx, output_Dual, h_data)
        optimizer_Dual.zero_grad()
        loss.backward()
        optimizer_Dual.step()
        for p in model_Dual.parameters():
            p.data = torch.max(torch.zeros_like(p.data), p.data)
            x1[iters - 1] = p.data[0][0]
            x2[iters - 1] = p.data[0][1]
            x3[iters - 1] = p.data[0][2]

        y2[iters - 1] = torch.mean(objective(output_Approx, h_data))
    #     t4 = time.time()
    #     print('train time: ', t4 - t3)

        model_Approx.eval()
        model_Dual.eval()
        with torch.no_grad():
            h_data = h_val_dataprovider.next()
            g_data = g_val_dataprovider.next()
            h_data, g_data = h_data.to(device), g_data.to(device)

            approx = model_Approx(h_data, g_data)
            y1[iters - 1] = torch.mean(objective(approx, h_data))

        if iters % show_interval == 0:
            print(iters, ' val: ', y1[iters - 1], ', train: ', y2[iters - 1])

        if iters % save_interval == 0:
            save_checkpoint({'state_approx_dict': model_Approx.state_dict(), 'state_dual_dict': model_Dual.state_dict(), 'val_cost': y1, 'train_cost': y2, 'lambda1': x1, 'lambda2': x2, 'lambda3': x3}, iters, tag='bnps-', path = save_path)
    #     t5 = time.time()
    #     print('eval time: ', t5 - t4)
    save_checkpoint({'state_approx_dict': model_Approx.state_dict(), 'state_dual_dict': model_Dual.state_dict(), 'val_cost': y1, 'train_cost': y2, 'lambda1': x1, 'lambda2': x2, 'lambda3': x3}, total_iters, tag='bnps-', path = save_path)


if __name__ == "__main__":
    main()