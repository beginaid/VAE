import os
import matplotlib.pyplot as plt
import numpy as np
import torch
# gif作成のためにmatplotlibのモジュールを利用
from matplotlib.animation import ArtistAnimation

class Visualize():
    def __init__(self, z_dim, dataloader_test, model, device):
        self.z_dim = z_dim
        self.dataloader_test = dataloader_test
        self.model = model
        self.device = device

    def createDirectories(self):
        if not os.path.exists("./images/reconstruction"):
            os.makedirs("./images/reconstruction")
        if not os.path.exists("./images/latent_space"):
            os.makedirs("./images/latent_space")
        if not os.path.exists("./images/lattice_point"):
            os.makedirs("./images/lattice_point")    
        if not os.path.exists("./images/walkthrough"):
            os.makedirs("./images/walkthrough")    


    def reconstruction(self):
        # テストデータを抽出
        count = 0
        for x, _ in self.dataloader_test:
            # testデータの先頭10個を利用する
            fig, axes = plt.subplots(2, 10, figsize=(20,4))
            # 画像の可視化のため軸とメモリは非表示に
            for i in range(axes.shape[0]):
                for j in range(axes.shape[1]): 
                    axes[i][j].set_xticks([])
                    axes[i][j].set_yticks([])
            for i, im in enumerate(x.view(-1, 28, 28)[:10]):
                axes[0][i].imshow(im, "gray")
            _, _, y = self.model(x, self.device)
            y = y.cpu().detach().numpy().reshape(-1, 28, 28)
            for i, im in enumerate(y[:10]):
                axes[1][i].imshow(im, "gray")
            fig.savefig(f"./images/reconstruction/model_z_{self.z_dim}_{count}.png")
            plt.close(fig)
            count += 1

    def latent_space(self):
        # colormapを準備
        cm = plt.get_cmap("tab10")
        # テストデータのバッチインデックスを表す変数
        count = 0
        # テストデータを抽出
        for x, t in self.dataloader_test:
            t = t.detach().numpy()
            # 各ラベルごとに可視化を行う
            fig_plot, ax_plot = plt.subplots(figsize=(9, 9))
            fig_scatter, ax_scatter = plt.subplots(figsize=(9, 9))
            _, z, _ = self.model(x, self.device)
            z = z.detach().numpy()
            for k in range(10):
                cluster_indexes = np.where(t==k)[0]
                ax_plot.plot(z[cluster_indexes,0], z[cluster_indexes,1], "o", ms=4, color=cm(k))
                ax_scatter.scatter(z[cluster_indexes,0], z[cluster_indexes,1], marker=f"${k}$", color=cm(k))
            fig_plot.savefig(f"./images/latent_space/model_z_{self.z_dim}_{count}_plot.png")
            fig_scatter.savefig(f"./images/latent_space/model_z_{self.z_dim}_{count}_scatter.png")
            plt.close(fig_plot)
            plt.close(fig_scatter)
            count += 1

    def lattice_point(self):
        # 一辺の生成画像数
        l = 25
        # 横軸と縦軸を設定して格子点を生成
        x = np.linspace(-2, 2, l)
        y = np.linspace(-2, 2, l)
        z_x, z_y = np.meshgrid(x, y)

        # 格子点を結合して潜在変数とみなす
        Z = torch.tensor(np.array([z_x, z_y]), dtype=torch.float).permute(1,2,0)
        # デコーダに潜在変数を入力
        y = self.model.decoder(Z).cpu().detach().numpy().reshape(-1, 28, 28)
        fig, axes = plt.subplots(l, l, figsize=(9, 9))
        # 可視化
        for i in range(l):
            for j in range(l):
                axes[i][j].set_xticks([])
                axes[i][j].set_yticks([])
                axes[i][j].imshow(y[l * (l - 1 - i) + j], "gray")
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.savefig(f"./images/lattice_point/model_z_{self.z_dim}.png")
        plt.close(fig)
        
    def walkthrough(z_dim, model):
        # 何枚の画像アニメーションに用いるか
        step = 50
        # 潜在変数は2
        z_dim = 2
        # 4つの方向のスタート地点とゴール地点を座標で定義
        z11 = torch.tensor([-3, 0], dtype=torch.float)
        z12 = torch.tensor([3, 0], dtype=torch.float)
        z21 = torch.tensor([-3, 3], dtype=torch.float)
        z22 = torch.tensor([3, -3], dtype=torch.float)
        z31 = torch.tensor([0, 3], dtype=torch.float)
        z32 = torch.tensor([0, -3], dtype=torch.float)
        z41 = torch.tensor([3, 3], dtype=torch.float)
        z42 = torch.tensor([-3, -3], dtype=torch.float)
        # for文を回すためにリスト化する
        z1_list = [z11, z21, z31, z41]
        z2_list = [z12, z22, z32, z42]
        # 線形変化させた潜在変数を格納するリスト
        z1_to_z2_list = []
        # デコーダの出力を格納するリスト
        y1_to_y2_list = []
        # 潜在変数のスタート地点からゴール地点を線形的に変化させてリストに格納する
        for z1, z2 in zip(z1_list, z2_list):
            z1_to_z2_list.append(torch.cat([((z1 * ((step - i) / step)) + (z2 * (i / step))) for i in range(step)]).reshape(step, z_dim))
        # 各潜在変数をデコーダに入力したときの出力をリストに格納する
        for z1_to_z2 in z1_to_z2_list:
            y1_to_y2_list.append(model.decoder(z1_to_z2).cpu().detach().numpy().reshape(-1, 28, 28))
        # gif化を行う
        for n in range(len(y1_to_y2_list)):
            fig, ax = plt.subplots(1, 1, figsize=(9,9))
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.set_xticks([])
            ax.set_yticks([])
            images = []
            for i, im in enumerate(y1_to_y2_list[n]):
                images.append([ax.imshow(im, "gray")])
            animation = ArtistAnimation(fig, images, interval=100, blit=True, repeat_delay=1000)
            animation.save(f"./images/walkthrough/linear_change_{n}.gif", writer="pillow")
            plt.close(fig)