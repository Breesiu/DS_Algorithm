import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
from utils import get_random_psd
import configs


class GMM2d():
    def __init__(self, params={}):
        self.n_components = 2
        self.params = params if params else self.initialize_random_params()

    def initialize_random_params(self):
        params = {'phi': np.random.uniform(0, 1),  # phi corresponds to the probability of the second gaussian
                  'mu0': np.random.normal(0, 1, size=(self.n_components,)),
                  'mu1': np.random.normal(0, 1, size=(self.n_components,)),
                  'sigma0': get_random_psd(self.n_components),
                  'sigma1': get_random_psd(self.n_components)}
        return params

    def get_pdf(self, x):
        return np.array(
            [(1 - self.params["phi"]) * (stats.multivariate_normal(self.params["mu0"], self.params["sigma0"]).pdf(x)), \
             (self.params["phi"]) * (stats.multivariate_normal(self.params["mu1"], self.params["sigma1"]).pdf(x))]).T

    def GMM_sklearn(self, x):
        model = GaussianMixture(n_components=2,
                                # variance_type='full',
                                tol=0.01,
                                max_iter=1000,
                                weights_init=[1 - self.params['phi'], self.params['phi']],
                                means_init=[self.params['mu0'], self.params['mu1']],
                                precisions_init=[self.params['sigma0'], self.params['sigma1']])
        model.fit(x)
        print("\nscikit learn:\n\tphi: %s\n\tmu_0: %s\n\tmu_1: %s\n\tsigma_0: %s\n\tsigma_1: %s"
              % (model.weights_[1], model.means_[0, :], model.means_[1, :], model.covariances_[0, :],
                 model.covariances_[1, :]))
        return model.predict(x), model.predict_proba(x)[:, 1]


class EM(GMM2d):
    def __init__(self, params={}):
        super().__init__(params)
        # print(params)co
        # self.q_0 = []
        # self.q_1 = []
        self.q_ = None
        self.unsupervised_log_likelihoods_history = []
    def e_step(self, x):
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        print(self.get_pdf(x))
        pdf = self.get_pdf(x)
        # for i in range(x.shape[0]):
        #
        #     pass
        # self.q_0 = (1-self.params["phi"])*pdf[:, 0] / ((1-self.params["phi"])*pdf[:, 0] + self.params["phi"]*pdf[:, 1])
        # print("q_0:{}, q_0.shape:{}".format(self.q_0, self.q_0.shape))
        # self.q_ = np.array(
        #     [(1-self.params["phi"])*pdf[:, 0] / ((1-self.params["phi"])*pdf[:, 0] + self.params["phi"]*pdf[:, 1]),
        #      (self.params["phi"])*pdf[:, 1] / ((1-self.params["phi"])*pdf[:, 0] + self.params["phi"]*pdf[:, 1])])
        self.q_ = np.array(
            [pdf[:, 0] / (pdf[:, 0] + pdf[:, 1]),
             pdf[:, 1] / (pdf[:, 0] + pdf[:, 1])])
        print("q_.shape:{}".format(self.q_.shape))
        # log_likelihood = 0
        # for j in range(x.shape[0]):
        #     log_likelihood += self.q_[0] * np.log()
        self.unsupervised_log_likelihoods_history.append((self.q_[0]*np.log(pdf[:, 0])+self.q_[1]*np.log(pdf[:, 1])).sum())
        print('logpdf_shape:{}'.format(np.log(pdf[:, 0]).shape))
        pass
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def m_step(self, x):
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        n = x.shape[0]
        # print(self.q_[0].T.shape)
        # print(x * self.q_[0, np.newaxis].T)
        q_1_sum = self.q_[1].sum()
        q_0_sum = self.q_[0].sum()
        phi = self.q_[1].mean()
        mu0 = (x * self.q_[0, np.newaxis].T).sum(0) / q_0_sum
        mu1 = (x * self.q_[1, np.newaxis].T).sum(0) / q_1_sum
        sigma0 = np.zeros((x.shape[1], x.shape[1]))
        sigma1 = np.zeros((x.shape[1], x.shape[1]))
        # print("x.shape:{}, mu0.shape:{}".format(x.shape, mu0))
        print((x[0]-mu0)[np.newaxis, :] @ (x[0]-mu0)[:, np.newaxis])
        for i in range(n):
            sigma0 += self.q_[0, i] * (x[i]-mu0)[:, np.newaxis] @ (x[i]-mu0)[np.newaxis, :]
        for i in range(n):
            sigma1 += self.q_[1, i] * (x[i]-mu1)[:, np.newaxis] @ (x[i]-mu1)[np.newaxis, :]
        print("x[i].shape:{},", (x[0]-mu0)[:, np.newaxis].shape)
        sigma0, sigma1 = sigma0/q_0_sum, sigma1/q_1_sum
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.params = {'phi': phi, 'mu0': mu0, 'mu1': mu1, 'sigma0': sigma0, 'sigma1': sigma1}
        print("params:{}".format(self.params))
        return self.params

    def run_em(self, x):
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # print(self.params)
        # phi corresponds to the probability of the second gaussian
        unsupervised_log_likelihoods = []
        for i in range(100):
            self.e_step(x)
            self.m_step(x)
        # predict
        predict = []
        unsupervised_posterior = []
        # unsupervised_log_likelihoods = []
        # for i in range(x.shape[0]):
        #
        #     pass
        unsupervised_posterior = self.q_;
        for i in range(x.shape[0]):
            if self.q_[0, i] > 0.5:
                predict.append(0)
            else:
                predict.append(1)
        pass
        return predict, unsupervised_posterior, self.unsupervised_log_likelihoods_history
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
