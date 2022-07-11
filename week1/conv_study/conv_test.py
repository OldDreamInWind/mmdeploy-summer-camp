import torch
import cv2
from torch import nn

def corr2d(X, K):  #@save
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

def conv(input, k, in_dim):
    if in_dim == 2:
        res = corr2d(input, k)
    elif in_dim == 3:
        _, _, c = input.shape
        h, w = k.shape
        res = torch.zeros((input.shape[0] - h + 1, input.shape[1] - w + 1, c))
        for i in range(c):
            res[:,:,i] = corr2d(input[:,:,i], k)
    return res


X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
ans = corr2d(X, K)
print('X = {}'.format(X))
print('K = {}'.format(K))
print('ans = {}'.format(ans))

img = cv2.imread("../image/screenshot.png")
img = torch.tensor(img)
kernel = torch.tensor([[1.0,0.0,-1.0],[1.0,0.0,-1.0],[1.0,0.0,-1.0]])
res = conv(img, kernel, 3)
cv2.imwrite("../image/result.png", res.numpy())



