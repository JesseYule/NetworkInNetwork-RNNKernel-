from data_preprocess import data_iter
import torch.optim as optim
import torch.nn as nn
import os
import torch
from RnnConv import rnnconv

datapath, trainset_name, validset_name, filetype = '../data', 'train_overfit.csv',\
                                                   'train_overfit.csv', 'csv'


train_iter, val_iter, TEXT, batchsize = data_iter(datapath, trainset_name, validset_name, filetype)

model = rnnconv(len(TEXT.vocab), 300, TEXT)

optimizer = optim.Adam(model.parameters(), lr=1e-6)

loss_fn = nn.CrossEntropyLoss()

# if os.path.exists('model_checkpoint/model.pkl'):
#     print('load model')
#     model.load_state_dict(torch.load('model_checkpoint/model.pkl'))

for epoch in range(100):
    train_acc = 0
    train_loss = 0
    min_loss = 1e5
    max_accuracy = 0

    for i, batch in enumerate(train_iter):

        optimizer.zero_grad()

        x = batch.Phrase
        y = batch.Sentiment

        preds = model(x)

        train_loss = loss_fn(preds, y)

        train_loss += train_loss.item()

        # if i % 10 == 0:
        print('train loss: ', train_loss.data)

        # 验证集
        if i % 1 == 0:
            val_acc = 0
            val_loss = 0
            val_batch = next(iter(val_iter))
            val_preds = model(val_batch.Phrase)
            _, result = torch.max(val_preds, 1)
            correct = 0
            for k in range(result.size()[0]):
                if result[k] - val_batch.Sentiment[k] == 0:
                    correct += 1
            accuracy = correct / result.size()[0]
            print('valid accuracy： ', accuracy)

            # if accuracy > max_accuracy:
            #     print('save model')
            #     torch.save(model.state_dict(), 'model_checkpoint/model.pkl')
            #     max_accuracy = accuracy

        train_loss.backward()
        optimizer.step()
