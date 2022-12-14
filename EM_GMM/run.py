import pandas as pd
from models import EM
from utils import learn_params
from matplotlib import pyplot as plt
import configs
import os

if __name__ == '__main__':
    # Read unlabelled data
    data_unlabeled = pd.read_csv(os.path.join(configs.data_dir, "unlabeled.csv"))
    x_unlabeled = data_unlabeled[["x1", "x2"]].values
    print(x_unlabeled)
    # Unsupervised learning
    print("unsupervised: ")
    em_unsupervised = EM()
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    unsupervised_forecasts, unsupervised_posterior, unsupervised_log_likelihoods = em_unsupervised.run_em(x_unlabeled)
    print("unsupervised_forecasts:{}, unsupervised_log_likelihoods:{}".format(unsupervised_forecasts, unsupervised_log_likelihoods))
    # visulization
    # plt.figure(dpi=300,figsize=(2,1) # 分辨率参数-dpi，画布大小参数-figsize

    fig, axs = plt.subplots(1, 3, figsize=(24, 16))
    # fig,axes = plt.subplots( 2,2, [figsize=(width,height)] )

    # plt.figure(figsize=(10, 10), dpi=500)
    column = ['unsupervised_log_likelihoods', 'Kmeans', 'GaussianMixture']
    for i in range(3):
        axs[i].set_title(column[i])
    axs[0].plot(unsupervised_log_likelihoods)
    axs[1].scatter(x_unlabeled[:, 0], x_unlabeled[:, 1], c=unsupervised_forecasts)
    plt.show()

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Read labelled data
    data_labeled = pd.read_csv(os.path.join(configs.data_dir, "labeled.csv"))
    x_labeled = data_labeled[["x1", "x2"]].values
    y_labeled = data_labeled["y"].values

    # Semi-supervised learning
    print("\nsemi-supervised: ")
    learned_params = learn_params(x_labeled, y_labeled)
    em_semisupervised = EM(params=learned_params)
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    semisupervised_forecasts, semisupervised_posterior, semisupervised_log_likelihoods = em_semisupervised.run_em(x_unlabeled)
    # visulization
    fig, axs = plt.subplots(1, 3, figsize=(24, 16))
    # fig,axes = plt.subplots( 2,2, [figsize=(width,height)] )
    # print("unsupervised_forecasts:{}, unsupervised_log_likelihoods:{}".format(semisupervised_forecasts, semisupervised_log_likelihoods))

    # plt.figure(figsize=(10, 10), dpi=500)
    column = ['unsupervised_log_likelihoods', 'Kmeans', 'GaussianMixture']
    for i in range(3):
        axs[i].set_title(column[i])
    axs[0].plot(semisupervised_log_likelihoods)
    axs[1].scatter(x_unlabeled[:, 0], x_unlabeled[:, 1], c=semisupervised_forecasts)
    plt.show()
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Compare the forecats with Scikit-learn API
    learned_params = learn_params(x_labeled, y_labeled)
    em_api = EM(params=learned_params)
    sklearn_forecasts, posterior_sklearn = em_api.GMM_sklearn(x_unlabeled)
    print("predict:", sklearn_forecasts)
    # print('{},{},{},{}'.format(semisupervised_forecasts.shape, semisupervised_posterior[:, 1],sklearn_forecasts.shape, posterior_sklearn.shape))
    output_df = pd.DataFrame({'semisupervised_forecasts': semisupervised_forecasts,
                              'semisupervised_posterior': semisupervised_posterior[1, :],
                              'sklearn_forecasts': sklearn_forecasts,
                              'posterior_sklearn': posterior_sklearn})

    print("\n%s%% of forecasts matched." % (
            output_df[output_df["semisupervised_forecasts"] == output_df["sklearn_forecasts"]].shape[0] /
            output_df.shape[0] * 100))
