__author__ = 'tracy'

from sklearn import svm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression

# Q1.1
df = pd.read_csv('/Users/tracy/msan-ml/hw-svm/svm-hw/P3.txt', sep=',', names=['X1', 'X2', 'y'])
train_X = np.array(df[['X1', 'X2']])
train_y = np.array(df[['y']]).reshape(len(df[['y']]))

clf = svm.SVC(kernel='linear')
clf.fit(train_X, train_y)
w = clf.coef_
w0 = clf.intercept_

print 'w' + str(clf.coef_)
print 'w0' + str(clf.intercept_)

# Q1.2
# create boundary
x1 = np.arange(-15.0, 30.0, 0.1)
x2 = -(w0 + w[0][0]*x1)/w[0][1]

# get index
pos_ind = np.where(train_y == 1)
neg_ind = np.where(train_y == -1)

# get the positive
train_X1_p = train_X[:, 0][pos_ind]
train_X2_p = train_X[:, 1][pos_ind]

# get the negative
train_X1_n = train_X[:, 0][neg_ind]
train_X2_n = train_X[:, 1][neg_ind]

# get support vector
sv = clf.support_vectors_
sv_X1 = sv[:, 0]
sv_X2 = sv[:, 1]
# support vector are labeled with black empty circle

# plot
plt.plot(x1, x2, 'b--', train_X1_p, train_X2_p, 'ro', train_X1_n, train_X2_n, 'go')
plt.plot(sv_X1, sv_X2, 'o', ms=10, mfc='none')
plt.show()


# Q1.3
clf_lda = LinearDiscriminantAnalysis(store_covariance=True)
clf_lda.fit(train_X, train_y)

# Q1.4
# MLE for mean and covariances with sklearn attributes
print clf_lda.means_   #[n_class, n_feature]
print clf_lda.covariance_ #[n_feature, n_feature]

# Derive decision boundary with log likelihood
w0 = - 0.5 * np.dot(np.dot(clf_lda.means_[0].T, np.linalg.inv(clf_lda.covariance_)), clf_lda.means_[0])\
         + 0.5 * np.dot(np.dot(clf_lda.means_[1].T, np.linalg.inv(clf_lda.covariance_)), clf_lda.means_[1])\
         + np.log(float(len(pos_ind[0]))/len(neg_ind[0]))
w = np.dot(np.linalg.inv(clf_lda.covariance_), ((clf_lda.means_[1]) - clf_lda.means_[0]))

slope = -w[0]/w[1]
intercept = -w0/w[1]
x_lda = np.arange(-15.0, 30, 0.1)
y_lda = slope * x_lda + intercept

# plot the boundary and compare it with svm
# the red line is svm boundary and the blue line is lda boundary
plt.plot(x_lda, y_lda, 'b--', x1, x2, 'r--', train_X1_p, train_X2_p, 'ro', train_X1_n, train_X2_n, 'go')
plt.show()

# Q1.5
# use P3_outlier to redo svm and lda again
######### SVM #########

df_out = pd.read_csv('/Users/tracy/msan-ml/hw-svm/svm-hw/P3_outlier.txt', sep=',')
train_X_out = np.array(df_out.ix[:, 0:-1])
train_y_out = np.array(df_out.ix[:, -1])

pos_ind_out = np.where(np.array(df_out.ix[:, -1]) == 1)
neg_ind_out = np.where(np.array(df_out.ix[:, -1]) == -1)

train_X_out_pos = train_X_out[pos_ind_out]
train_X_out_neg = train_X_out[neg_ind_out]

# train with SVM
clf_out = svm.SVC(kernel='linear')
clf_out.fit(train_X_out, train_y_out)

# SVM boundary
x_out_boundary_1 = np.arange(-20.0, 50.0, 0.1)
x_out_boundary_2 = -(clf_out.intercept_ + clf_out.coef_[0][0]*x_out_boundary_1)/clf_out.coef_[0][1]

# plot
plt.plot(x_out_boundary_1, x_out_boundary_2, 'b--', train_X_out_pos[:, 0], train_X_out_pos[:, 1], 'ro',
         train_X_out_neg[:, 0], train_X_out_neg[:, 1], 'go')
plt.show()


######### LDA #########

# train with LDA
lda_out = LinearDiscriminantAnalysis(store_covariance=True)
lda_out.fit(train_X_out, train_y_out)

# LDA boundary
# Derive decision boundary with log likelihood
w0_out = - 0.5 * np.dot(np.dot(lda_out.means_[0].T, np.linalg.inv(lda_out.covariance_)), lda_out.means_[0])\
         + 0.5 * np.dot(np.dot(lda_out.means_[1].T, np.linalg.inv(lda_out.covariance_)), lda_out.means_[1])\
         + np.log(float(len(pos_ind_out[0]))/len(neg_ind_out[0]))
w_out = np.dot(np.linalg.inv(lda_out.covariance_), ((lda_out.means_[1]) - lda_out.means_[0]))

slope = -w_out[0]/w_out[1]
intercept = -w0_out/w_out[1]
x_lda_out = np.arange(-20.0, 50.0, 0.1)
y_lda_out = slope * x_lda_out + intercept

# plot
plt.plot(x_lda_out, y_lda_out, 'b--', train_X_out_pos[:, 0], train_X_out_pos[:, 1], 'ro',
         train_X_out_neg[:, 0], train_X_out_neg[:, 1], 'go')
plt.show()

# Q1.6
# The result when there is outlier. SVM performs much better than LDA. This is reasonable because
# LDA requires the data set satisfying the gaussian distribution. But in the case of outlier, it's
# hard to be satisfied.

# Q1.7
logit_clf = LogisticRegression()
logit_clf.fit(train_X, train_y)
logit_x = np.arange(-15.0, 30, 0.1)
logit_y = -(logit_clf.coef_[0][0]*logit_x + logit_clf.intercept_)/logit_clf.coef_[0][1]

# plot decision boundary of logistic and SVM and the P3 data
plt.plot(x1, x2, 'b--', train_X1_p, train_X2_p, 'ro', train_X1_n, train_X2_n, 'go', logit_x, logit_y, 'b-')
plt.show()


#######################################

# Q2.1
df_2 = pd.read_csv('/Users/tracy/msan-ml/hw-svm/svm-hw/P4_train.txt', sep=',', header=False)
P4_train_X = np.array(df_2.ix[:, 0:-1])
P4_train_y = np.array(df_2.ix[:, -1])

df_2_test = pd.read_csv('/Users/tracy/msan-ml/hw-svm/svm-hw/P4_test.txt', sep=',')
P4_test_X = np.array(df_2_test.ix[:, 0:-1])
P4_test_y = np.array(df_2_test.ix[:, -1])

# K_fold
kf = KFold(n=len(P4_train_X), n_folds=5)
clf_quaK = svm.SVC(C=1, gamma=0.1, kernel='poly', degree=2)
clf_quaK.fit(P4_train_X, P4_train_y)
print np.mean(clf_quaK.predict(P4_test_X) == P4_test_y)

# quadratic kernel
# get the optimal C and gamma for quadratic kernel
max_accuracy_quaK = 0
for C in np.arange(0.1, 10, 1):
    print C
    for gamma in np.arange(0.01, 1, 0.5):
        temp = []
        for train_index, test_index in kf:
            clf_quaK = svm.SVC(C=C, gamma=gamma, kernel='poly', degree=2)
            clf_quaK.fit(P4_train_X[train_index], P4_train_y[train_index])
            accuracy = np.mean(clf_quaK.predict(P4_train_X[test_index]) == P4_train_y[test_index])
            temp.append(accuracy)
        if np.mean(temp) > max_accuracy_quaK:
            opt_C_quaK = C
            opt_gamma_quaK = gamma

# gaussian kernel
# get the optimal C and gamma for gaussian kernel
max_accuracy_rbf = 0
for C in np.arange(0.1, 10, 1):
    print C
    for gamma in np.arange(0.01, 1, 0.5):
        temp = []
        for train_index, test_index in kf:
            clf_rbfK = svm.SVC(C=C, gamma=gamma)
            clf_rbfK.fit(P4_train_X[train_index], P4_train_y[train_index])
            accuracy = np.mean(clf_rbfK.predict(P4_train_X[test_index])==P4_train_y[test_index])
            temp.append(accuracy)
        if np.mean(temp) > max_accuracy_rbf:
            opt_C_rbfK = C
            opt_gamma_rbfK = gamma

# Compare quaK and rbfK. Train with the whole set of train data set and use test data set to test accuracy
clf_quaK = svm.SVC(C=opt_C_quaK, gamma=opt_gamma_quaK, kernel='poly', degree=2)
clf_quaK.fit(P4_train_X, P4_train_y)
accuracy_quaK = np.mean(clf_quaK.predict(P4_test_X) == P4_test_y)

clf_rbfK = svm.SVC(C=opt_C_rbfK, gamma=opt_gamma_rbfK)
clf_rbfK.fit(P4_train_X, P4_train_y)
accuracy_rbfK = np.mean(clf_rbfK.predict(P4_test_X) == P4_test_y)

print "The accuracy of quadratic kernel is " + str(accuracy_quaK)  #0.843601895735
print "The accuracy of gaussian kernel is " + str(accuracy_rbfK)   #0.820269777616

# The accuracy of quadratic kernel is higher than gaussian kernel.


# Q2.2
# grid on K to get the optimal K with highest accuracy
K_list = []
accuracy_neigh_list = []
max_accuracy_KNN = 0
for K in range(1, 10):
    neigh = KNeighborsClassifier(n_neighbors=K)
    neigh.fit(P4_train_X, P4_train_y)
    accuracy_neigh = np.mean(neigh.predict(P4_test_X)==P4_test_y)
    if max_accuracy_KNN < accuracy_neigh:
        opt_K = K

# train the model the optimal K and test
neigh = KNeighborsClassifier(n_neighbors=opt_K)
neigh.fit(P4_train_X, P4_train_y)
accuracy_neigh = np.mean(neigh.predict(P4_test_X)==P4_test_y)

# compare accuracy_quaK, accuracy_rbfK, accuracy_neigh
print "The accuracy of KNN is " + str(accuracy_neigh)   #0.824401506866

# Q2.3
# The accuracy of KNN is very similar to gaussian kernel. And it performs a little better than gaussian kernel SVM.
# But KNN doesn't perform better than quadratic kernel SVM.
# All the methods are basically good at handling high dimensional data.








