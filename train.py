# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import dataset
from model import CNN
from evaluate import main as evaluate

num_epochs = 100
batch_size = 32
learning_rate = 0.0001


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = CNN().to(torch.device("cuda"))
    cnn.train()

    criterion = nn.MultiLabelSoftMarginLoss().to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    max_eval_acc = -1
    
    train_dataloader = dataset.get_train_data_loader(batch_size=batch_size)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.float().to(device)
            predict_labels = cnn(images)
            loss = criterion(predict_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print("Epoch:", epoch + 1, "Step:", i, "Loss:", loss.item())
            if (i + 1) % 100 == 0:
                # current is model.pkl
                torch.save(cnn.state_dict(), "./model.pkl")
                print("Save model")
        print("Epoch:", epoch, "Step:", i, "Loss:", loss.item())
        eval_acc = evaluate()
        if eval_acc > max_eval_acc:
            # best model save as best_model.pkl
            torch.save(cnn.state_dict(), "./best_model.pkl")
            print("Save best model")
    torch.save(cnn.state_dict(), "./model.pkl")
    print("Save last model")


if __name__ == '__main__':
    main()
