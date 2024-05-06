import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.mlab as mlab
from Colors import Color
import pandas as pd
import scipy.stats as stats # 统计学库
from scipy.stats import norm  # 用于拟合正态分布曲线
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 16})

file_path = r'residual_data.xlsx'
df = pd.read_excel(file_path)
cnn = df['cnn'].to_numpy().astype('float32')
attention = df['attention'].to_numpy().astype('float32')
lstm = df['lstm'].to_numpy().astype('float32')
lstm_sa = df['lstm_sa'].to_numpy().astype('float32')

lenged_list = ['CNN','Attention','LSTM','LSTM-SA']


df_sentivity = pd.read_excel(r'dataset/random_2.xlsx')
hiddensize_4 = df_sentivity['hiddensize=4'].to_numpy().astype('float32')
hiddensize_8 = df_sentivity['hiddensize=8'].to_numpy().astype('float32')
hiddensize_10 = df_sentivity['hiddensize=10'].to_numpy().astype('float32')
hiddensize_12 = df_sentivity['hiddensize=12'].to_numpy().astype('float32')
hiddensize_24 = df_sentivity['hiddensize=24'].to_numpy().astype('float32')
hiddensize_32 = df_sentivity['hiddensize=32'].to_numpy().astype('float32')
hiddensize_48 = df_sentivity['hiddensize=48'].to_numpy().astype('float32')
hiddensize_64 = df_sentivity['hiddensize=64'].to_numpy().astype('float32')
hiddensize_128 = df_sentivity['hiddensize=128'].to_numpy().astype('float32')
hiddensize_256 = df_sentivity['hiddensize=256'].to_numpy().astype('float32')
# def hist():
#
#
#     plt.hist(cnn,bins=80,density=True,alpha=0.95,histtype='stepfilled',color=Color.CNN )
#     plt.hist(attention,bins=80,density=True,alpha=0.90,histtype='stepfilled',color=Color.Attention )
#     plt.hist(lstm, bins=80, density=True, alpha=0.85, histtype='stepfilled', color=Color.LSTM)
#     plt.hist(lstm_sa, bins=80, density=True, alpha=0.75, histtype='stepfilled', color=Color.LSTM_SA)
#     plt.legend(lenged_list)
#
#     plt.show()
#     # plt.savefig('final.png', dpi=2000, bbox_inches='tight')
#     return None

# def kdeplot():
#
#     sns.kdeplot(cnn,shade=False,color=Color.CNN)
#
#     # sns.displot(cnn, color=Color.CNN)
#     plt.show()
#     # plt.savefig('final.png', dpi=2000, bbox_inches='tight')
#     return None

def normal_fit():
    # x = [i for i in range(100)]


    # x = np.linspace(mu - 3*sigma, mu + 3*sigma, 50)

    # num_bins = 30  # 直方图柱子的数量
    # n, bins, patches = plt.hist(cnn, num_bins, normed=1, facecolor='blue', alpha=0.5)
    # y = mlab.normpdf(bins, mu, sigma)
    ###############    hidden_size 4   ###################
    mu, sigma = np.mean(hiddensize_4), np.std(hiddensize_4)
    x = np.linspace(-1.5, 1, 200)
    # sns.kdeplot()
    y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    plt.plot(x, y)  # 绘制y的曲线



    # ###############    CNN   ###################
    # mu, sigma = np.mean(cnn), np.std(cnn)
    # x = np.linspace(-1.5, 1,200)
    # # sns.kdeplot()
    # y = np.exp(-(x - mu) ** 2 /(2* sigma **2))/(math.sqrt(2*math.pi)*sigma)
    # plt.plot(x, y, color=Color.CNN)  # 绘制y的曲线
    #
    # ###############    Attention   ###################
    # mu, sigma = np.mean(attention), np.std(attention)
    # x = np.linspace(-1.5, 1,200)
    # # sns.kdeplot()
    # y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    # plt.plot(x, y, color=Color.Attention)  # 绘制y的曲线
    #
    # ###############    LSTM   ###################
    # mu, sigma = np.mean(lstm), np.std(lstm)
    # x = np.linspace(-1.5, 1,200)
    # # sns.kdeplot()
    # y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    # plt.plot(x, y, color=Color.LSTM)  # 绘制y的曲线
    #
    # ###############    LSTM-SA   ###################
    # mu, sigma = np.mean(lstm_sa), np.std(lstm_sa)
    # x = np.linspace(-1.5, 1,200)
    # # sns.kdeplot()
    # y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    # plt.plot(x, y, color=Color.LSTM_SA)  # 绘制y的曲线

    # plt.legend(lenged_list)


    plt.show()
    # plt.savefig('final.png', dpi=2000, bbox_inches='tight')
    return mu,sigma

def normal_fit_sentivity():
    # x = [i for i in range(100)]


    # x = np.linspace(mu - 3*sigma, mu + 3*sigma, 50)

    # num_bins = 30  # 直方图柱子的数量
    # n, bins, patches = plt.hist(cnn, num_bins, normed=1, facecolor='blue', alpha=0.5)
    # y = mlab.normpdf(bins, mu, sigma)

    linewidth = 0.75
    alpha = 0.3
    left = -2.0
    right = 2.0
    ###############    hidden_size 4   ###################
    mu, sigma = np.mean(hiddensize_4), np.std(hiddensize_4)
    x = np.linspace(left, right, 200)
    # sns.kdeplot()
    y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    plt.plot(x, y,linewidth=linewidth,alpha=alpha)  # 绘制y的曲线

    ###############    hidden_size 8   ###################
    mu, sigma = np.mean(hiddensize_8), np.std(hiddensize_8)
    x = np.linspace(left, right, 200)
    # sns.kdeplot()
    y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    plt.plot(x, y,linewidth=linewidth,alpha=alpha)  # 绘制y的曲线


    ###############    hidden_size 10   ###################
    mu, sigma = np.mean(hiddensize_10), np.std(hiddensize_10)
    x = np.linspace(left, right, 200)
    # sns.kdeplot()
    y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    plt.plot(x, y,linewidth=linewidth,alpha=alpha)  # 绘制y的曲线


    ###############    hidden_size 12   ###################
    mu, sigma = np.mean(hiddensize_12), np.std(hiddensize_12)
    x = np.linspace(left, right, 200)
    # sns.kdeplot()
    y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    plt.plot(x, y,linewidth=linewidth,alpha=alpha)  # 绘制y的曲线


    ###############    hidden_size 24   ###################
    mu, sigma = np.mean(hiddensize_24), np.std(hiddensize_24)
    x = np.linspace(left, right, 200)
    # sns.kdeplot()
    y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    plt.plot(x, y,linewidth=linewidth,alpha=alpha)  # 绘制y的曲线


    ###############    hidden_size 32   ###################
    mu, sigma = np.mean(hiddensize_32), np.std(hiddensize_32)
    x = np.linspace(left, right, 200)
    # sns.kdeplot()
    y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    plt.plot(x, y,linewidth=linewidth,alpha=alpha)  # 绘制y的曲线


    ###############    hidden_size 48   ###################
    mu, sigma = np.mean(hiddensize_48), np.std(hiddensize_48)
    x = np.linspace(left, right, 200)
    # sns.kdeplot()
    y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    plt.plot(x, y,linewidth=linewidth,alpha=alpha)  # 绘制y的曲线


    ###############    hidden_size 64   ###################
    mu, sigma = np.mean(hiddensize_64), np.std(hiddensize_64)
    x = np.linspace(left, right, 200)
    # sns.kdeplot()
    y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    plt.plot(x, y,linewidth=linewidth,alpha=alpha)  # 绘制y的曲线


    ###############    hidden_size 128   ###################
    mu, sigma = np.mean(hiddensize_128), np.std(hiddensize_128)
    x = np.linspace(left, right, 200)
    # sns.kdeplot()
    y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    plt.plot(x, y,linewidth=linewidth,alpha=alpha)  # 绘制y的曲线


    ###############    hidden_size 256   ###################
    mu, sigma = np.mean(hiddensize_256), np.std(hiddensize_256)
    x = np.linspace(left, right, 200)
    # sns.kdeplot()
    y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    plt.plot(x, y,linewidth=linewidth,alpha=alpha)  # 绘制y的曲线



    # ###############    CNN   ###################
    # mu, sigma = np.mean(cnn), np.std(cnn)
    # x = np.linspace(-1.5, 1,200)
    # # sns.kdeplot()
    # y = np.exp(-(x - mu) ** 2 /(2* sigma **2))/(math.sqrt(2*math.pi)*sigma)
    # plt.plot(x, y, color=Color.CNN)  # 绘制y的曲线
    #
    # ###############    Attention   ###################
    # mu, sigma = np.mean(attention), np.std(attention)
    # x = np.linspace(-1.5, 1,200)
    # # sns.kdeplot()
    # y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    # plt.plot(x, y, color=Color.Attention)  # 绘制y的曲线
    #
    # ###############    LSTM   ###################
    # mu, sigma = np.mean(lstm), np.std(lstm)
    # x = np.linspace(-1.5, 1,200)
    # # sns.kdeplot()
    # y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    # plt.plot(x, y, color=Color.LSTM)  # 绘制y的曲线
    #
    # ###############    LSTM-SA   ###################
    # mu, sigma = np.mean(lstm_sa), np.std(lstm_sa)
    # x = np.linspace(-1.5, 1,200)
    # # sns.kdeplot()
    # y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    # plt.plot(x, y, color=Color.LSTM_SA)  # 绘制y的曲线

    plt.legend()


    # plt.show()
    plt.savefig('random_2灵敏度误差分布.pdf', dpi=2000, bbox_inches='tight')
    return mu,sigma

# def normal_fit_ok():
#     norm_comparision_plot(cnn)
#
# def norm_comparision_plot(data, figsize=(12, 10), color="#099DD9",
#                           ax=None, surround=True, grid=True):
#     """
#     function: 传入 DataFrame 指定行，绘制其概率分布曲线与正态分布曲线(比较)
#     color: 默认为标准天蓝  #F79420:浅橙  ‘green’：直接绿色(透明度自动匹配)
#     ggplot 经典三原色：'#F77B72'：浅红, '#7885CB'：浅紫, '#4CB5AB'：浅绿
#     ax=None: 默认无需绘制子图的效果；  surround：sns.despine 的经典组合，
#                                          默认开启，需要显式关闭
#     grid：是否添加网格线，默认开启，需显式关闭
#     """
#     plt.figure(figsize=figsize) # 设置图片大小
#     # fit=norm: 同等条件下的正态曲线(默认黑色线)；lw-line width 线宽
#     # sns.distplot(data, fit=norm, color=color,kde_kws={"color" :color, "lw" :3 }, ax=ax)
#     (mu, sigma) = norm.fit(data)  # 求同等条件下正态分布的 mu 和 sigma
#     # 添加图例：使用格式化输入，loc='best' 表示自动将图例放到最合适的位置
#     plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)] ,loc='best')
#     plt.ylabel('Frequency')
#     plt.title("Distribution")
#     if surround == True:
#         # trim=True-隐藏上面跟右边的边框线，left=True-隐藏左边的边框线
#         # offset：偏移量，x 轴向下偏移，更加美观
#         sns.despine(trim=True, left=True, offset=10)
#     if grid == True:
#         plt.grid(True)  # 添加网格线
#
#
#     plt.show()

if __name__ == '__main__':
    # hist()
    print(normal_fit_sentivity())
    # normal_fit_ok() # 不行