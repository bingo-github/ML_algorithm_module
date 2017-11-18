# -*- coding: utf-8 -*-
# 梯度下降法的矩阵求解
import random
import numpy as np


def d_J(X, Y, theta):
    # X表示样本
    # Y表示样本对应的标签
    # theta表示参数
    return np.matmul(X.T, (np.matmul(X, theta) - Y)) / np.shape(X)[0]


def cost_J(X, Y, theta):
    # X表示样本
    # Y表示样本对应的标签
    # theta表示参数
    X_theta_red_Y = np.matmul(X, theta) - Y
    return np.matmul(X_theta_red_Y.T, X_theta_red_Y) / (2 * np.shape(X)[0])


step_size = 0.001  # 步长
max_iters = 10000  # 最大迭代次数
eps = 0.0001  # 精度


def train_gadient_descent(X, Y, theta):
    cost = 100
    cur_iters = 0
    while cost > eps and cur_iters < max_iters:
        theta = theta - step_size * d_J(X, Y, theta)
        cur_iters += 1
        cost = cost_J(X, Y, theta)
    return theta


if __name__ == '__main__':
    # 输入的特征和标签
    X = np.array([[1, 4], [2, 5], [5, 1], [4, 2]])  # feature
    Y = np.array([[19], [26], [19], [20]])  # lebal
    # 假设的函数为 y=theta0+theta1*x1+theta2*x2
    theta = np.array([[0.1], [0.1]])  # 初始的theta参数
    print(train_gadient_descent(X, Y, theta))
