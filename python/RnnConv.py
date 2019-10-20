import torch
import torch.nn as nn
import numpy as np


class rnnconv(nn.Module):
    def __init__(self, vocab_size, embedding_dim, TEXT):
        super(rnnconv, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # nn.init.xavier_uniform_(self.embedding.weight) # 初始化权重
        self.embedding.weight.data.copy_(TEXT.vocab.vectors)  # 载入预训练词向量

        self.LSTM_stack = nn.LSTM(embedding_dim, hidden_size=300, num_layers=1, batch_first=True,
                                  bidirectional=True)

        # 文本处理或许用一维卷积就足够了，因为文本的向量是二维（词向量和句子向量），不像图像可能是三维
        # 卷积运算针对完整一个词向量在句子方向上进行计算，核的大小为kernel_size*len(word_emb)，不确定如果在词向量方向也进行多次卷积是否有意义
        self.rnnkernel = nn.Conv1d(in_channels=600, out_channels=600, kernel_size=1)

        self.transform11 = nn.Conv1d(in_channels=600, out_channels=600, kernel_size=1)
        self.transform12 = nn.Conv1d(in_channels=600, out_channels=600, kernel_size=1)

        self.transform21 = nn.Conv1d(in_channels=600, out_channels=600, kernel_size=1)
        self.transform22 = nn.Conv1d(in_channels=600, out_channels=600, kernel_size=1)

        self.mlp1 = nn.Linear(600*48, 1000)
        self.mlp2 = nn.Linear(1000, 5)

    def forward(self, x):
        x = x.permute(1, 0)

        x = self.embedding(x)

        # 对两个输入序列用BiLSTM分析上下文含义，重新进行embedding
        x, _ = self.LSTM_stack(x.float())  # (batch, sentence_len, hidden_units)

        # 整个矩阵乘同一个数可以保留词向量之间的关系并放大它们之间的差异
        norm_const_x = torch.abs(torch.reciprocal_(torch.mean(x)))
        x_norm = x * norm_const_x
        x_norm = x_norm.permute(0, 2, 1)

        # 假设RNN卷积核尺寸为3，即三个词为一组进行处理，用indice选择时刻1、2、3对应的数据
        indices1 = torch.tensor(range(0, 48))
        indices2 = torch.tensor(range(1, 49))
        indices3 = torch.tensor(range(2, 50))

        x_norm1 = torch.index_select(x_norm, 2, indices1)
        x_norm2 = torch.index_select(x_norm, 2, indices2)
        x_norm3 = torch.index_select(x_norm, 2, indices3)

        h1 = self.rnnkernel(x_norm1)
        h1 = self.transform11(h1)
        x_norm2 = self.transform12(x_norm2)
        h2 = self.rnnkernel(x_norm2 + h1)
        h2 = self.transform21(h2)
        x_norm3 = self.transform22(x_norm3)
        output = self.rnnkernel(x_norm3 + h2)

        output = output.view(100, 600*48)

        output = self.mlp1(output.squeeze(0))
        output = torch.relu(output)
        output = self.mlp2(output)
        output = torch.sigmoid(output)

        return output
