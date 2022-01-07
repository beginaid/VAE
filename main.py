import os
import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from models.VAE import VAE
from libs.Visualize import Visualize

class Main():
    def __init__(self, z_dim):
        self.z_dim = z_dim
        self.dataloader_train
        self.dataloader_valid
        self.dataloader_test
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPUが付か合える場合はGPU上で動かす
        self.model = VAE(self.z_dim).to(self.device) # VAEクラスのコンストラクタに潜在変数の次元数を渡す
        self.writer = SummaryWriter(log_dir="./logs") # tensorboardでモニタリングする
        self.lr = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr) # 今回はoptimizerとしてAdamを利用
        self.num_epochs = 3 # 最大更新回数は1000回
        self.num_no_improved = 0 # early stoppingを判断するためのフラグ
        self.loss_valid_min = 10 ** 7 # 各種lossを格納しておくリスト
        self.num_batch_train = 0
        self.num_batch_valid = 0
        self.Visualize = Visualize(self.z_dim, self.dataloader_test, self.model, self.device)


    def createDirectories(self):
        if not os.path.exists("./logs"):
            os.makedirs("./logs")
        if not os.path.exists("./params"):
            os.makedirs("./params")

    def createDataLoader(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]) # MNISTのデータをとってくるときに一次元化する前処理
        dataset_train_valid = datasets.MNIST("./", train=True, download=True, transform=transform) # trainデータとtestデータに分けてデータセットを取得
        dataset_test = datasets.MNIST("./", train=False, download=True, transform=transform)

        # trainデータの20%はvalidationデータとして利用
        size_train_valid = len(dataset_train_valid) # 60000
        size_train = int(size_train_valid * 0.8) # 48000
        size_valid = size_train_valid - size_train # 12000
        dataset_train, dataset_valid = torch.utils.data.random_split(dataset_train_valid, [size_train, size_valid])

        # 取得したデータセットをDataLoader化する
        self.dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1000, shuffle=True)
        self.dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1000, shuffle=False)
        self.dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1000, shuffle=False)

    def train_batch(self):
        self.model.train()
        for x, _ in self.dataloader_train:
            lower_bound, _, _ = self.model(x, self.device) # VAEにデータを流し込む
            loss = -sum(lower_bound) # lossは負の下限
            self.model.zero_grad() # 訓練時のpytorchのお作法
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar("Loss_train/KL", -lower_bound[0].cpu().detach().numpy(), self.num_iter + self.num_batch) # 各種lossをtensorboardに格納
            self.writer.add_scalar("Loss_train/Reconst", -lower_bound[1].cpu().detach().numpy(), self.num_iter + self.num_batch)
            self.num_batch += 1
        self.num_batch -= 1

    def valid_batch(self):
        loss = []
        self.model.eval()
        for x, _ in self.dataloader_valid:
            lower_bound, _, _ = self.model(x, self.device) # VAEにデータを流し込む
            loss.append(-sum(lower_bound).cpu().detach().numpy()) # 各種lossをリストに格納
            self.writer.add_scalar("Loss_valid/KL", -lower_bound[0].cpu().detach().numpy(), self.num_iter + self.num_batch) # 各種lossをtensorboardに格納
            self.writer.add_scalar("Loss_valid/Reconst", -lower_bound[1].cpu().detach().numpy(), self.num_iter + self.num_batch)
            self.num_batch += 1    
        self.num_batch -= 1
        self.loss_valid = np.mean(loss)
        self.loss_valid_min = np.minimum(self.loss_valid_min, self.loss_valid)

    def early_stopping(self, loss_valid):
        if self.loss_valid_min < loss_valid: # もし今までのlossの最小値よりも今回のイテレーションのlossが大きければカウンタ変数をインクリメントする
            self.num_no_improved += 1
            print(f"{self.num_no_improved}回連続でValidationが悪化しました")
        else: # もし今までのlossの最小値よりも今回のイテレーションのlossが同じか小さければカウンタ変数をリセットする
            self.num_no_improved = 0
            torch.save(self.model.state_dict(), f"./params/model_z_{self.z_dim}.pth")

    def main(self):
        self.createDirectories()
        self.createDataLoader()    
        print("-----学習開始-----")
        for self.num_iter in range(self.num_epochs):
            self.train_batch()
            self.valid_batch()
            print(f"[EPOCH{self.num_iter + 1}] loss_valid: {int(self.loss_valid)} | Loss_valid_min: {int(self.loss_valid_min)}") # 各種lossを出力
            self.early_stopping(self.loss_valid)
            if self.num_no_improved >= 10:
                print(f"{self.num_no_improved}回連続でValidationが悪化")
                break
        print("-----学習完了-----")
        # tensorboardのモニタリングも停止しておく
        self.writer.close()
        
        # 評価対象のモデルをインスタンス化する
        self.model.load_state_dict(torch.load(f"./params/model_z_{self.z_dim}.pth"))
        self.model.eval()
        self.Visualize.createDirectories()
        self.Visualize.reconstruction()
        self.Visualize.latent_space()
        self.Visualize.lattice_point()
        self.Visualize.walkthrough()
    
if __name__ == '__main__':
    main = Main(2)
    main.main()