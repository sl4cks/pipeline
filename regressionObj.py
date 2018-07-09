import numpy as np
from abc import ABC, abstractmethod
from sklearn import linear_model
import sklearn.metrics.pairwise as pr
import sklearn.kernel_ridge as kr
from scipy.linalg import eig

#NOTES
# -Need to review the specific behavior of super() and what is needed from it
# -Probably don't need to store the labels on the LDA implementation

def geig(A, B):
    [U,lam,V] = np.linalg.svd(B) #Do svd on B, get three matrices but V = U.T
    lam = np.diag(np.sqrt(lam)**-1)
    aTil = lam @ U.T @ A @ U @ lam #because U should be symmetrical
    aTil = (aTil+aTil.T)/2 #make sure A~ is symmetriccal

    l,w = eig(aTil)
    w = np.real(w)
    l = np.real(l)
    v = U @ lam @ w
    return l,v

# #A is the dataset, x is estimated solution, b is results
# def findError(A, x, b):
#     return np.linalg.norm((A @ x) - b)

def createTheta(thetaSize, noise=0, linear=False):
    if linear:
        theta = np.random.uniform(-1, 1, (thetaSize, 1))
        X = np.concatenate((np.random.randn(thetaSize, noise), theta, 2*theta), axis=1).T
        # X = np.concatenate((np.random.randn(thetaSize, noise),
        #     theta+np.random.randn(thetaSize, 1)/10, 2*theta+np.random.randn(thetaSize, 1)/10), axis=1).T
        return X,theta
    else:
        theta = 2 * np.pi * np.random.rand(thetaSize, 1)
        X = np.concatenate((np.cos(theta), np.sin(theta), np.random.randn(thetaSize, noise)), axis=1).T
        return X,theta

def createEllipse(arrLen, noiseDen=20):
    #create an appropriately sized vector of zeros and ones
    ones = np.ones((1, arrLen))
    nones = np.zeros((1, arrLen))

    #create three sets of data, each with a 1 on a respective axis
    #fourth dimension is pure noise, so it stays 0 for now
    x = np.concatenate((ones, nones, nones, nones), axis=0)
    y = np.concatenate((nones, ones, nones, nones), axis=0)
    z = np.concatenate((nones, nones, ones, nones), axis=0)

    #Add just a little noise to each of the sets of data
    x = x + np.random.randn(4, arrLen)/noiseDen
    y = y + np.random.randn(4, arrLen)/noiseDen
    z = z + np.random.randn(4, arrLen)/noiseDen
    cat = np.concatenate((x, y, z), axis=1)
    labels = np.squeeze(np.concatenate((ones, ones*2, ones*3), axis=1))
    return cat,labels

def createImages(N, n=40):
    [X,Y] = np.meshgrid(range(0,n), range(0,n))

    xs = np.ceil(n*np.random.rand(N, 1))
    ys = np.ceil(n*np.random.rand(N, 1))

    A = np.zeros((n, n, N))

    for i in range(1, N+1):
        A[:,:,i-1] = (np.absolute(X-xs[i-1]) < 3) & (np.absolute(Y-ys[i-1]) < 3)

    A = np.reshape(A,(n**2,N)) #turn images into big column vectors
    mu = np.mean(A, 1, keepdims=True) #center the data
    A = A - np.tile(mu,(1, N))
    xs = xs - np.mean(xs)
    return A,xs

class Data(object):
    def __init__(self, A):
        #possible check size
        self.A = A #make this a numpy array object

    @property
    def A(self):
        return self.A

    @A.setter
    def A(self, value):
        #need to check stuff here
        self.A = value

    @A.deleter
    def A(self):
        del self.A

class SupData(Data):
    def __init__(self, A, b=None):
        #check for size mismatches
        self.b = b
        super().__init__(A)

    @property
    def b(self):
        return self.b

    @b.setter
    def b(self, value):
        #need to check size with A here
        self.b = value

    @b.deleter
    def b(self):
        del self.b

class DimensionReduction(ABC):
    def __init__(self):
        self.proj = None

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def reduce(self):
        pass

class SLDA(DimensionReduction):
    def __init__(self):
        self.labels = None
        self._eigen_vectors = None
        self._eigen_values = None
        self._thresholds = None
        self.bins = None

    #return an array of the inverse sort of an argsort
    def inv_argsort(self, ind):
        n = np.size(ind)
        arr = np.zeros(n, dtype='int')
        arr[ind] = range(n)
        return arr

    #might want to make this a static method
    def create_labels(self, n, bins):
        labels = np.zeros(n)

        #create the set of labels to use
        for i in range(n):
            labels[i] = (i//(n//bins))

        #if everything doesn't fit evenly, throw last values into last bin
        if n%bins != 0:
            labels[n-n%bins:n] = bins-1

        return labels

    def train(self, data, sol, bins=3):
        self.bins = bins
        n = np.size(sol)
        labels = self.create_labels(n, bins)

        #this gets the sorted indices from the solution, the arranges the data and solution by them
        inds = np.squeeze(np.argsort(sol, axis=0))
        pinv = self.inv_argsort(inds)
        labels = labels[pinv]

        self._thresholds = np.zeros(bins-1)
        self._thresholds = sol[inds][n//bins:bins*(n//bins):n//bins]

        categoryMeans = np.zeros((np.size(data, axis=0), bins))
        for i in range(bins):
            #Find all the indices with label i
            iinds = np.where(labels==i)

            #Find the mean of the data points with the i-th label
            categoryMeans[:,i] = np.mean(np.squeeze(data[:,iinds]), axis=1)

            #Subtract the i-th mean from each data point with label i
            data[:,iinds] = data[:,iinds] - np.tile(np.reshape(categoryMeans[:,i],(
                np.size(categoryMeans[:,i]),1,1)),(1,1,np.size(iinds)))

        sigmab = np.cov(categoryMeans)
        sigma = np.cov(np.squeeze(data))


        [L,U] = geig(sigmab,sigma)

        sinds = np.argsort(abs(L))
        sinds = sinds[::-1]
        L = L[sinds] #Sort the eigenvalues
        U = U[:,sinds]

        for col in range(np.size(U ,axis=0)):
            U[:,col] = U[:,col]/np.linalg.norm(U[:,col])

        self._eigen_vectors = U
        self._eigen_values = L
        self.labels = labels

    def reduce(self, data, sol=None):
        if sol is None:
            return self._eigen_vectors.T @ data
        else:
            sol = np.squeeze(sol)
            labels = np.ones(np.size(sol)) * -1
            for i in range(np.size(self._thresholds)):
                thresh = self._thresholds[i]
                #may need to be greater than or equal to
                labels[np.where((thresh > sol) * (labels == -1))] = i

            labels[np.where(labels == -1)] = np.size(self._thresholds)

            return (self._eigen_vectors.T @ data)[0:self.bins-1,:], labels

class Regression(ABC):
    def __init__(self):
        #initialize solution to None
        self.x = None

    @abstractmethod
    def regress(self):
        pass

    #make this implementation a linearregression child object
    @abstractmethod
    def map(self, A):
        return A @ self.x

class KernelRidge(Regression):
    def __init__(self):
        self._X_fit = None
        super().__init__()

    def regress(self, A, b, alpha=1, kernel='rbf'):
        if kernel is 'rbf':
            ker = kr.KernelRidge(alpha, kernel="rbf")
        else:
            print("whoops, this isn't implemented")

        ker.fit(A, b)
        self.x = ker.dual_coef_
        self._X_fit = ker.X_fit_

    def map(self, A):
        return

class LeastSquares(Regression):
    def __init__(self):
        #resulting rank, residuals, and singular values from least
        #a least squares regression. x is declared None in super
        self._rank = None
        self._residuals = None
        self._s = None
        super().__init__()

    def regress(self, A, b, rcond=None):
        self.x,self.residuals,self.rank,self.s = np.linalg.lstsq(
            A, b, rcond=rcond)

class Lasso(Regression):
    def __init__(self):
        self.sparse_x = None
        super().__init__()

    #Need to implement the other arguments that are all in the lasso
    def regress(self, A, b, alpha=1.0, max_iter=1000):
        lasso = linear_model.Lasso(alpha=alpha, max_iter=max_iter)
        lasso.fit(A, b)
        #lasso returns an n size array rather than nx1 vector
        self.x = np.reshape(lasso.coef_, (np.size(lasso.coef_),1))
        self._sparse_x = lasso.sparse_coef_

    def sparse_map(self, A):
        return A @ self.sparse_x

class Ridge(Regression):
    def __init__(self):
        self.sparse_x = None
        super().__init__()

    #Need to implement the other arguments that are all in the lasso
    #Default params are the following
    #(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver=’auto’, random_state=None
    def regress(self, A, b, alpha=1.0, max_iter=1000):
        ridge = linear_model.Ridge(alpha=alpha, max_iter=max_iter)
        ridge.fit(A, b)
        #ridge returns a 1x2 instead of a 2x1 vector so we store the transpose
        self.x = ridge.coef_.T

class Threshold(Regression):
    def regress(self, A, b, thresh=0.05):
        valid = range(np.size(A, axis=1))
        finalX = np.zeros((np.size(A, axis=1),1))
        unfit = [1] #initialized to one so while loop triggers

        #iterate until no more values fall below the threshold
        while np.size(unfit) > 0 and np.size(A) > 0:
            #do the least squares to find a new x solution
            [x,res,rank,s] = np.linalg.lstsq(A, b, rcond=None)

            #get indices of values below threshold
            unfit = np.where(abs(x) < thresh)

            #delete unfit values from master range and A for next iteration
            A = np.delete(A, unfit, axis=1)
            valid = np.delete(valid, unfit)

        np.put(finalX, valid, x)
        self.x = finalX

#a version of the threshold method that drops the n lowest values i times from the set of values
class ThresholdIter(Regression):
    def regress(self, A, b, n, i):
        valid = range(np.size(A, axis=1))
        finalX = np.zeros((np.size(A, axis=1),1))
        for j in range(i):
            if(np.size(A) == 0):
                break

            [x,res,rank,s] = np.linalg.lstsq(A, b, rcond=None)
            #there may be a faster way to do this with partitions but eh
            index = np.squeeze(x).argsort()[:n]
            A = np.delete(A, index, axis=1)
            valid = np.delete(valid, index)

        np.put(finalX, valid, x)
        return finalX
