# -*- coding: UTF-8 -*-
import numpy as np
import torch
import setting
import dataset
import encoding as ohe
from model import CNN


def main():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = CNN().to(device)
    cnn.eval()
    cnn.load_state_dict(torch.load('best_model.pkl', weights_only=True))
    print("load cnn net.")
    
    predict_dataloader = dataset.get_predict_data_loader()


    for i, (images, labels) in enumerate(predict_dataloader):
        image = images.to(device)
        predict_label = cnn(image)
        
        c0 = setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:setting.ALL_CHAR_SET_LEN].cpu().data.numpy())]
        c1 = setting.ALL_CHAR_SET[np.argmax(predict_label[0, setting.ALL_CHAR_SET_LEN:2 * setting.ALL_CHAR_SET_LEN].cpu().data.numpy())]
        c2 = setting.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * setting.ALL_CHAR_SET_LEN:3 * setting.ALL_CHAR_SET_LEN].cpu().data.numpy())]
        c3 = setting.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * setting.ALL_CHAR_SET_LEN:4 * setting.ALL_CHAR_SET_LEN].cpu().data.numpy())]
        
        c = '%s%s%s%s' % (c0, c1, c2, c3)
        x = ohe.decode(np.squeeze(labels.numpy()))
        s = True if c == x else False
        print(f'Predictive Label: {c}, Actual Label: {x}, Result: {s}')


if __name__ == '__main__':
    main()
