import torch
import torch.nn as nn


class nin(nn.Module):
    def __init__(self, vocab_size, embedding_dim, TEXT):
        super(nin, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # nn.init.xavier_uniform_(self.embedding.weight) # 初始化权重
        self.embedding.weight.data.copy_(TEXT.vocab.vectors)  # 载入预训练词向量

        self.LSTM_stack = nn.LSTM(embedding_dim, hidden_size=300, num_layers=1, batch_first=True,
                                  bidirectional=True)

        # 文本处理或许用一维卷积就足够了，因为文本的向量是二维（词向量和句子向量），不像图像可能是三维
        # 卷积运算针对完整一个词向量在句子方向上进行计算，核的大小为kernel_size*len(word_emb)，不确定如果在词向量方向也进行多次卷积是否有意义

        self.nin = nn.Sequential(
            nn.Conv1d(in_channels=600, out_channels=800, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=800, out_channels=400, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=400, out_channels=300, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5),

            nn.Conv1d(in_channels=150, out_channels=200, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=200, out_channels=200, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=200, out_channels=200, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5),

            nn.Conv1d(in_channels=100, out_channels=300, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=300, out_channels=300, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=300, out_channels=5, kernel_size=1),
            nn.ReLU(inplace=True),
            # nn.AvgPool2d(kernel_size=4, stride=1, padding=0),
            nn.Linear(8, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)

        )

    def forward(self, x):
        x = x.permute(1, 0)
        x = self.embedding(x)

        # 对两个输入序列用BiLSTM分析上下文含义，重新进行embedding
        x, _ = self.LSTM_stack(x.float())  # (batch, sentence_len, hidden_units)

        # 整个矩阵乘同一个数可以保留词向量之间的关系并放大它们之间的差异
        norm_const_x = torch.abs(torch.reciprocal_(torch.mean(x)))
        x_norm = x * norm_const_x
        x_norm = x_norm.permute(0, 2, 1)
        out = self.nin(x_norm)
        out = out.squeeze(2)

        return out
