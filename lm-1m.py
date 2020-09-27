# -> copy from https://www.ethanrosenthal.com/2016/01/09/
# explicit-matrix-factorization-sgd-als/
import numpy as np
import pandas as pd
import copy
np.random.seed(0)


names = ['user_id', 'item_id', 'rating', 'tomestamp']
df = pd.read_table('text/ml-1m/rating_test.dat', sep="::", names=names,
                   engine='python')
n_users = df.user_id.unique().shape[0]
item_ids = list(df.item_id.unique())
n_items = len(item_ids)
ratings = np.zeros((n_users, n_items))

for row in df.itertuples():
    ratings[row[1]-1, item_ids.index(row[2])] = row[3]  # 评分矩阵

# Split into training and test sets.
# Remove 10 ratings for each user
# and assign them to the test set
# why not remove 100 user?


def train_test_split(ratings):
    test = np.zeros(ratings.shape)  # 零矩阵 测试集合
    train = copy.deepcopy(ratings)
    for user in range(ratings.shape[0]):  # 选取一个用户值不为0的十个数字
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0],
                                        size=10,
                                        replace=True)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
# Test and training are truly disjoint
    assert(np.all((train * test) == 0))  # 判断是否成功
    return train, test


train, test = train_test_split(ratings)
# -  - - - -  - - - - - - - - - - - - - - - -


from sklearn.metrics import mean_squared_error  # 均方误差 E(t-t')^2

def get_mse(pred, actual):  # √
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()  # 返回一个以为数组
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


# - - - - - - - - IMF - - - - - ----------
from numpy.linalg import solve


class ExplicitMF():  # 实现隐式矩阵因式分解
    def __init__(self,
                 ratings,
                 n_factors=40,
                 item_reg=0.0,
                 user_reg=0.0,
                 verbose=False):
        """
        Train a matrix factorization model to predict empty
                entries in a matrix. The terminology assumes a
        ratings matrix which is ~ user x item

        Params
        ======
        ratings : (ndarray)
            User x Item matrix with corresponding ratings

        n_factors : (int)
            Number of latent factors to use in matrix
            factorization model

        item_reg : (float)
            Regularization term for item latent factors

        user_reg : (float)
            Regularization term for user latent factors

        verbose : (bool)
            Whether or not to printout training progress
        """

        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.item_reg = item_reg
        self.user_reg = user_reg
        self._v = verbose

    def als_step(self,
                 latent_vectors,
                 fixed_vecs,
                 ratings,
                 _lambda,
                 type='user'):  # √ 公式推导 dd
        """
        One of the two ALS steps. Solve for the latent vectors
        specified by type.
        """
        if type == 'user':
            # Precompute
            YTY = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(YTY.shape[0]) * _lambda

            for u in range(latent_vectors.shape[0]):
                latent_vectors[u, :] = solve((YTY + lambdaI),
                                             ratings[u, :].dot(fixed_vecs))
        elif type == 'item':
            # Precompute
            XTX = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(XTX.shape[0]) * _lambda

            for i in range(latent_vectors.shape[0]):
                latent_vectors[i, :] = solve((XTX + lambdaI),
                                             ratings[:, i].T.dot(fixed_vecs))
        return latent_vectors

    def train(self, n_iter=10):  # √
        """ Train model for n_iter iterations from scratch."""
        # initialize latent vectors 初始化拟合矩阵
        self.user_vecs = np.random.random((self.n_users, self.n_factors))
        self.item_vecs = np.random.random((self.n_items, self.n_factors))

        self.partial_train(n_iter)

    def partial_train(self, n_iter):  # √
        """
        Train model for n_iter iterations. Can be
        called multiple times for further training.
        """
        ctr = 1
        while ctr <= n_iter:
            if ctr % 10 == 0 and self._v:
                print('\tcurrent iteration: {}'.format(ctr))
            self.user_vecs = self.als_step(self.user_vecs,
                                           self.item_vecs,
                                           self.ratings,
                                           self.user_reg,
                                           type='user')
            self.item_vecs = self.als_step(self.item_vecs,
                                           self.user_vecs,
                                           self.ratings,
                                           self.item_reg,
                                           type='item')
            ctr += 1

    def predict_all(self): # √ 生成predict矩阵
        """ Predict ratings for every user and item. """
        predictions = np.zeros((self.user_vecs.shape[0],
                                self.item_vecs.shape[0]))
        for u in range(self.user_vecs.shape[0]):
            for i in range(self.item_vecs.shape[0]):
                predictions[u, i] = self.predict(u, i)

        return predictions

    def predict(self, u, i): # √
        """ Single user and item prediction. """
        return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)

    def calculate_learning_curve(self, iter_array, test):
        """
        Keep track of MSE as a function of training iterations.

        Params
        ======
        iter_array : (list)
            List of numbers of iterations to train for each step of
            the learning curve. e.g. [1, 5, 10, 20]
        test : (2D ndarray)
            Testing dataset (assumed to be user x item).

        The function creates two new class attributes:

        train_mse : (list)
            Training data MSE values for each value of iter_array
        test_mse : (list)
            Test data MSE values for each value of iter_array
        """
        iter_array.sort()
        self.train_mse = []
        self.test_mse = []
        iter_diff = 0  # √
        for (i, n_iter) in enumerate(iter_array):
            if self._v:
                print('Iteration: {}'.format(n_iter))
            if i == 0:
                self.train(n_iter - iter_diff)  # 迭代次数
            else:
                self.partial_train(n_iter - iter_diff)

            predictions = self.predict_all()

            self.train_mse += [get_mse(predictions, self.ratings)]
            self.test_mse += [get_mse(predictions, test)]
            if self._v:
                print('Train mse: ' + str(self.train_mse[-1]))
                print('Test mse: ' + str(self.test_mse[-1]))
            iter_diff = n_iter

# - - - - - - - - try - - - - - -  - - -
MF_ALS = ExplicitMF(train, n_factors=15,
                    user_reg=-1.0, item_reg=-1.0)
iter_array = [1, 2, 5, 10, 25, 50]
MF_ALS.calculate_learning_curve(iter_array, test)

print("OK")

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def plot_learning_curve(iter_array, model):
    plt.plot(iter_array, model.train_mse,
             label='Training', linewidth=5)
    plt.plot(iter_array, model.test_mse,
             label='Test', linewidth=5)


plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('iterations', fontsize=30)
plt.ylabel('MSE', fontsize=30)
plt.legend(loc='best', fontsize=20, labels=['train', 'text'])
plot_learning_curve(iter_array, MF_ALS)
plt.show()
# - - - - - - - - 选参 - - - - - - - - -
# latent_factors = [5, 10, 20, 40, 80]
# regularizations = [0.01, 0.1, 1., 10., 100.]
# regularizations.sort()
# iter_array = [1, 2, 5, 10, 25, 50, 100]

# best_params = {}
# best_params['n_factors'] = latent_factors[0]
# best_params['reg'] = regularizations[0]
# best_params['n_iter'] = 0
# best_params['train_mse'] = np.inf
# best_params['test_mse'] = np.inf
# best_params['model'] = None

# for fact in latent_factors:
#     print('Factors: {}'.format(fact))
#     for reg in regularizations:
#         print('Regularization: {}'.format(reg))
#         MF_ALS = ExplicitMF(train, n_factors=fact,
#                             user_reg=reg, item_reg=reg)
#         MF_ALS.calculate_learning_curve(iter_array, test)
#         min_idx = np.argmin(MF_ALS.test_mse)
#         if MF_ALS.test_mse[min_idx] < best_params['test_mse']:
#             best_params['n_factors'] = fact
#             best_params['reg'] = reg
#             best_params['n_iter'] = iter_array[min_idx]
#             best_params['train_mse'] = MF_ALS.train_mse[min_idx]
#             best_params['test_mse'] = MF_ALS.test_mse[min_idx]
#             best_params['model'] = MF_ALS
#             print('New optimal hyperparameters')
#             print(pd.Series(best_params))


