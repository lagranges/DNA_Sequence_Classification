__author__ = 'vle020518'

import numpy as np
import cvxopt
from cvxopt import matrix, solvers
class Kernel_SVM(object):

    # kernel in {"gaussian","linear","poly"}
    def __init__(self, kernel):
        self.kernel = kernel
        self._c = 1.2

    def _generate_predictor(self,X,y,lambdas):
        sv_indices = lambdas > 1e-6

        support_multipliers = lambdas[sv_indices]
        support_vectors = X[sv_indices]
        support_vector_labels = np.transpose(y)[sv_indices]

        bias = np.mean(
            [y_k - SVMPredictor(
                kernel=self._kernel,
                bias=0.0,
                weights=support_multipliers,
                support_vectors=support_vectors,
                support_vector_labels=support_vector_labels).predict(np.array([x_k]))[0]
             for (y_k, x_k) in zip(support_vector_labels, support_vectors)])

        return SVMPredictor(
            kernel=self._kernel,
            bias=bias,
            weights=support_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels)


    # train
    def fit(self,X,y):
        n_samples, n_features = X.shape
        if self.kernel == "poly":
            self._kernel = self._poly()
        elif self.kernel == "linear":
            self._kernel = self._linear()
        else:
            self._kernel = self._gaussian(1/n_features)
        # compute lambdas using CVXOPT for Quadratic Programming
        lambdas = self._compute_lambdas(X,y)
        return self._generate_predictor(X,y, lambdas)

    # Solves
        # min 1/2 x^T P x + q^T x
        # s.t.
        #  Gx \coneleq h
        # P = ynymK
    #  Ax = b
    def _compute_lambdas(self,X,y):
        n_samples, n_features = X.shape
        K = self._compute_Kmatrix(X)

        P = matrix(np.outer(y,y)*K)
        #q = matrix(-1 * np.ones(n_samples))
        q = matrix(-np.ones((n_samples, 1)))

        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))

        # a_i \leq c
        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)

        G = matrix(-np.eye(n_samples)) # for all lambda_n >= 0
        h = matrix(np.zeros((n_samples, 1)))

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        A = matrix(y)
        b = matrix(np.zeros((1, 1)))
        solvers.options['show_progress'] = False
        solution = solvers.qp(P,q,G,h,A,b)
        return np.ravel(solution['x'])

    def _gaussian(self,sigma):
        def f(x,y):
            return np.exp(-sigma*np.linalg.norm(x-y) ** 2)
        return f

    def _linear(self):
        def f(x,y):
            return np.inner(x,y)
        return f

    def _poly(self):
        def f(x,y):
            return np.inner(np.inner(x,y)**2)
        return f

    def _compute_Kmatrix(self, X):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples,n_samples))
        #K =
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i][j] = self._kernel(x_i,x_j)
                #K[i][j] = np.inner(x_i,x_j)
        return K

    def predict(self,features):
        # sign(x.w+b)
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)

        return classification

        #print(self._compute_Kmatrix(X))
        self._compute_lambdas(X,y)

class SVMPredictor(object):
    def __init__(self,
                 kernel,
                 bias,
                 weights,
                 support_vectors,
                 support_vector_labels):
        self._kernel = kernel
        self._bias = bias
        self._weights = weights
        self._support_vectors = support_vectors
        self._support_vector_labels = support_vector_labels

    def predict(self, X):
        """
        Computes the SVM prediction on the given features x.
        """
        n_samples, n_features = X.shape
        y = np.zeros((1,n_samples))
        for j,x_it in enumerate(X):
            r = self._bias
            for z_i, x_i, y_i in zip(self._weights,
                                 self._support_vectors,
                                 self._support_vector_labels):
                r += z_i * y_i * self._kernel(x_i, x_it)
            y[0][j] = np.sign(r).item()
        return y

    def score(self,X,y):
        predicted = self.predict(X)
        count = 0
        for i, y_it in enumerate(np.transpose(y)):
            if y_it == predicted[0][i]:
                count += 1

        return count/y.shape[1]