import numpy as np


def get_random_psd(n):
    x = np.random.normal(0, 1, size=(n, n))
    return np.dot(x, x.transpose())


def learn_params(x_labeled, y_labeled):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # print('x_labeled_shape:{}, y_labeled_shape:{},x_labeled:{},y_labeled:{}'.format(x_labeled.shape, y_labeled.shape,x_labeled, y_labeled))
    n = x_labeled.shape[0]
    # print(self.q_[0].T.shape)
    # print(x * self.q_[0, np.newaxis].T)
    q_1_sum = y_labeled.sum()
    q_0_sum = n-q_1_sum
    phi = y_labeled.mean()
    mu0 = (x_labeled * (1-y_labeled)[:, np.newaxis]).sum(0) / q_0_sum
    mu1 = (x_labeled * y_labeled[:, np.newaxis]).sum(0) / q_1_sum
    sigma0 = np.zeros((x_labeled.shape[1], x_labeled.shape[1]))
    sigma1 = np.zeros((x_labeled.shape[1], x_labeled.shape[1]))
    # print("x.shape:{}, mu0.shape:{}".format(x.shape, mu0))
    # print((x[0] - mu0)[np.newaxis, :] @ (x[0] - mu0)[:, np.newaxis])
    for i in range(n):
        sigma0 += (1-y_labeled)[i] * (x_labeled[i] - mu0)[:, np.newaxis] @ (x_labeled[i] - mu0)[np.newaxis, :]
    for i in range(n):
        sigma1 += y_labeled[i] * (x_labeled[i] - mu1)[:, np.newaxis] @ (x_labeled[i] - mu1)[np.newaxis, :]
    # print("x[i].shape:{},", (x[0] - mu0)[:, np.newaxis].shape)
    sigma0, sigma1 = sigma0 / q_0_sum, sigma1 / q_1_sum
    pass
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    params = {'phi': phi, 'mu0': mu0, 'mu1': mu1, 'sigma0': sigma0, 'sigma1': sigma1}
    return params
