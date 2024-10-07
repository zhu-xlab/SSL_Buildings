import os
import json
import torch
import torch.nn as nn
import numpy as np
import os.path as osp
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import norm, bernoulli

from dataloader import CreateTestDataLoader
from algorithms import create_algorithm 
from options.test_options import TestOptions
from utils import Evaluator, update_soft_conf_matrix, upsample_size, upsample_scale
from utils import unnormalize
import cv2

torch.set_num_threads(4)
torch.cuda.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_norms(fore_norm, back_norm, logits, labels, lbl_fore=1, lbl_back=0, momentum=0.9):
    samples_fore = logits[labels==lbl_fore]
    samples_back = logits[labels==lbl_back]

    if samples_fore.shape[0] > 0:
        samples_fore_mean = np.mean(samples_fore)
        samples_fore_std = np.std(samples_fore)
        fore_norm['mean'] = fore_norm['mean'] * momentum + samples_fore_mean * (1 - momentum)
        fore_norm['std'] = fore_norm['std'] * momentum + samples_fore_std * (1 - momentum)

    if samples_back.shape[0] > 0:
        samples_back_mean = np.mean(samples_back)
        samples_back_std = np.std(samples_back)
        back_norm['mean'] = back_norm['mean'] * momentum + samples_back_mean * (1 - momentum)
        back_norm['std'] = back_norm['std'] * momentum + samples_back_std * (1 - momentum)
    return fore_norm, back_norm


def plot_gaussian_fit_and_save(fore_norm, back_norm, bins=50, filename=None):
    """
    绘制两组数据（前景和背景）的直方图和高斯拟合曲线，并在提供文件名时保存图像。

    参数:
        fore_norm (dict): 包含 'mean' 和 'std' 的前景数据字典。
        back_norm (dict): 包含 'mean' 和 'std' 的背景数据字典。
        bins (int, optional): 直方图的箱数。默认值为 50。
        filename (str, optional): 保存图像的路径。如果为 None，则显示图像。
    """
    # 根据均值和标准差生成数据
    fore_data = np.random.normal(fore_norm['mean'], fore_norm['std'], 1000)
    back_data = np.random.normal(back_norm['mean'], back_norm['std'], 1000)

    # 设置画布和子图
    plt.figure(figsize=(10, 6))

    # 绘制前景数据的直方图和高斯拟合曲线
    y1, x1, _ = plt.hist(fore_data, bins=bins, alpha=0.5, color='blue', density=True, label='Building')
    mu1, std1 = fore_norm['mean'].item(), fore_norm['std'].item()
    p1 = norm.pdf(np.linspace(min(fore_data), max(fore_data), 100), mu1, std1)
    plt.plot(np.linspace(min(fore_data), max(fore_data), 100), p1, 'b--', linewidth=2)
    plt.axvline(mu1, color='blue', linestyle='dashed', linewidth=1)
    plt.text(mu1, max(p1), f'Mean: {mu1:.2f}', color='blue')

    # 绘制背景数据的直方图和高斯拟合曲线
    y2, x2, _ = plt.hist(back_data, bins=bins, alpha=0.5, color='red', density=True, label='Background')
    mu2, std2 = back_norm['mean'].item(), back_norm['std'].item()
    p2 = norm.pdf(np.linspace(min(back_data), max(back_data), 100), mu2, std2)
    plt.plot(np.linspace(min(back_data), max(back_data), 100), p2, 'r--', linewidth=2)
    plt.axvline(mu2, color='red', linestyle='dashed', linewidth=1)
    plt.text(mu2, max(p2), f'Mean: {mu2:.2f}', color='red')

    # 设置图例和标题
    plt.legend()
    plt.title('Histogram and Gaussian Fit with CDF Difference')
    plt.xlabel('Value')
    plt.ylabel('Density / Absolute Difference')

    # 判断是否需要保存图像
    if filename:
        plt.savefig(filename, format='png', dpi=300)  # 保存图像到指定的文件路径
        plt.close()  # 关闭图像，防止再次显示
        print(f"Image saved as {filename}")
    else:
        # 显示图形
        plt.show()


def test(fore_norm, back_norm, test_loader, algorithm, logit_mode, num_classes):
    num_classes = 2
    if 'DBMatch' in algorithm.name:
        print (algorithm.bias)

    algorithm.eval()
    metric = Evaluator(num_classes)
    with torch.no_grad():
        for i, (image, label, _) in enumerate(test_loader):
            image, label = image.cuda(), label.cuda()  
            feat = algorithm.get_feats(image)
            logit = algorithm.get_logits(feat)
            if 'DBMatch' in algorithm.name:
                logit = algorithm.debias_logits(logit)
            prob = torch.sigmoid(logit)

            prob = prob.reshape(-1).cpu().detach().numpy()            
            logit = logit.reshape(-1).cpu().detach().numpy()
            label = label.reshape(-1).cpu().detach().numpy()
            pred = (prob > 0.5).astype(np.int64)
            metric.add_batch(label, pred)

            if logit_mode == 'GT':
                fore_norm, back_norm = update_norms(fore_norm, back_norm, logit, label)
            elif logit_mode == 'Pred':
                fore_norm, back_norm = update_norms(fore_norm, back_norm, logit, pred)

    test_Precision, test_Recall = metric.Precision_Recall()
    test_IoU = metric.Intersection_over_Union()
    test_F1 = metric.F1_Score()
    return fore_norm, back_norm, test_Recall[1], test_Precision[1], test_IoU[1], test_F1[1]


if __name__ == '__main__':    
    opt = TestOptions()
    args = opt.initialize()
    args.percent = str(int(args.percent)) if args.percent >= 1 else str(args.percent) 
    args.save_dir = os.path.join(args.save_dir, args.dataset.replace('/', '_'), args.model)
    opt.print_options(args)

    # build loader, build algorithm, and load checkpoint
    test_loader = CreateTestDataLoader(args)    
    algorithm = create_algorithm(args, device)
    save_checkpoint_path = os.path.join(args.save_dir, '%s_p-%s_%s_0.6' % \
                           (args.dataset.replace('/', '_'), args.percent, args.algorithm) + '.pth')
    if os.path.exists(save_checkpoint_path):
        print ('loading checkpoint from {}'.format(save_checkpoint_path))
        resume = torch.load(save_checkpoint_path)
        algorithm.load_state_dict(resume['state_dict'], strict=False)
    else:
        raise FileNotFoundError(f"Error: The file '{save_checkpoint_path}' does not exist.")      

    # test epoch
    logit_mode = 'GT'    # [GT, Pred]
    fore_norm = {'mean': np.zeros(1), 'std': np.ones(1)}
    back_norm = {'mean': np.zeros(1), 'std': np.ones(1)}
    fore_norm, back_norm, test_Rec, test_Prec, test_IoU, test_F1 \
            = test(fore_norm, back_norm, test_loader, algorithm, logit_mode, args.num_classes)

    # print class-wise logit distributions
    plot_gaussian_fit_and_save(
        fore_norm, back_norm, filename='./outputs/{}_{}_{}%_{}_ema_classes.png'.format(
            args.dataset.replace('/', '_'), args.algorithm, args.percent, logit_mode))

    print ('Test on {}'.format(args.dataset))
    print (' Rec, Pre, IoU, F1 : & {} & {} & {} & {}'.format(
        np.round(test_Rec*100, 2), np.round(test_Prec*100, 2), 
        np.round(test_IoU*100, 2), np.round(test_F1*100, 2)))

