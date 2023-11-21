import numpy as np
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
### TODO: import any other packages you need for your solution
import cvxpy as cp

#--- Task 1 ---#
class MyClassifier:  
    def __init__(self, K):
        self.K = K  # number of classes

        ### TODO: Initialize other parameters needed in your algorithm
        # examples:
        # self.w = None
        # self.b = None

    
    def train(self, trainX):
        ''' Task 1-2 
            TODO: train classifier using LP(s) and updated parameters needed in your algorithm 
        '''
        
    
    def predict(self, testX):
        ''' Task 1-2 
            TODO: predict the class labels of input data (testX) using the trained classifier
        '''


        # Return the predicted class labels of the input data (testX)
        return predY
    

    def evaluate(self, testX, testY):
        predY = self.predict(testX)
        accuracy = accuracy_score(testY, predY)

        return accuracy
    

##########################################################################
#--- Task 2 ---#
class MyClustering:
    def __init__(self, K):
        self.K = K  # number of classes
        self.labels = None

        ### TODO: Initialize other parameters needed in your algorithm
        # examples: 
        self.cluster_centers_ = None
        
    
    def train(self, trainX):
        ''' Task 2-2 
            TODO: cluster trainX using LP(s) and store the parameters that discribe the identified clusters
        '''

        # STEP1 - initialization: randomly select K points from trainX (initial centroids)
        n = trainX.shape[0]  # number of data points
        m = trainX.shape[1]  # size of each data point
        self.cluster_centers_ = trainX[np.random.choice(n, size=self.K, replace=False)]

        # STEP2 - iteration of assignment and update
        itr = 0
        max_itr = 1e6
        trh = 1e-6
        while itr < max_itr:
            # iteration
            itr = itr + 1

            # distance d_ij
            vector_from_centers = trainX.reshape(n, 1, m) - self.cluster_centers_  # n x K x m
            distance_from_centers = np.linalg.norm(vector_from_centers, axis=2)  # n x K
            d_ij = distance_from_centers.flatten()

            # STEP2.1 - Assignment (Linear Programming)
            # constraint 1: each data has to be assigned to one cluster
            A_1 = np.zeros((n, n * self.K))
            for i in range(n):
                A_1[i, i * self.K: (i + 1) * self.K] = np.ones(self.K)
            b_1 = np.ones(n)
            # constraint 2: a cluster includes at least one data
            A_2 = np.tile(np.eye(self.K), n)
            b_2 = np.ones(self.K)
            # constraint 3: integer
            A_3 = np.eye(n * self.K)
            b_3 = np.zeros(n * self.K)
            A_4 = np.eye(n * self.K)
            b_4 = np.ones(n * self.K)
            # variable x_ij
            x_ij = cp.Variable(n * self.K)
            # optimization problem
            prob = cp.Problem(cp.Minimize(d_ij.T @ x_ij),
                              [A_1 @ x_ij == b_1,
                               A_2 @ x_ij >= b_2,
                               A_3 @ x_ij >= b_3,
                               A_4 @ x_ij <= b_4])
            prob.solve()
            # assign labels
            x_ij = x_ij.value.reshape((n, self.K))
            new_labels = np.argmax(x_ij, axis=1)

            # update cluster centroids
            new_cluster_centers_ = np.zeros_like(self.cluster_centers_)
            for j in np.arange(self.K):
                new_cluster_centers_[j, :] = np.mean(trainX[new_labels == j], axis=0)

            # convergence criteria
            if np.max(np.linalg.norm(self.cluster_centers_ - new_cluster_centers_, axis=1)) < trh:
                self.cluster_centers_ = new_cluster_centers_
                self.labels = new_labels
                break
            else:
                self.cluster_centers_ = new_cluster_centers_
                self.labels = new_labels
            # if np.all(new_labels == self.labels):
            #     break
            # else:
            #     self.labels = new_labels

        # Update and return the cluster labels of the training data (trainX)
        return self.labels
    
    
    def infer_cluster(self, testX):
        ''' Task 2-2 
            TODO: assign new data points to the existing clusters
        '''

        # distance of points from centroids n x K
        vector_from_centers = testX.reshape(testX.shape[0], 1, testX.shape[1]) - self.cluster_centers_  # n x K x m
        distance_from_centers = np.linalg.norm(vector_from_centers, axis=2)  # n x K
        pred_labels = np.argmin(distance_from_centers, axis=1)

        # Return the cluster labels of the input data (testX)
        return pred_labels
    

    def evaluate_clustering(self, trainY):
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(self.labels, label_reference)
        nmi = normalized_mutual_info_score(trainY, aligned_labels)

        return nmi
    

    def evaluate_classification(self, trainY, testX, testY):
        pred_labels = self.infer_cluster(testX)
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(pred_labels, label_reference)
        accuracy = accuracy_score(testY, aligned_labels)

        return accuracy


    def get_class_cluster_reference(self, cluster_labels, true_labels):
        ''' assign a class label to each cluster using majority vote '''
        label_reference = {}
        for i in range(len(np.unique(cluster_labels))):
            index = np.where(cluster_labels == i,1,0)
            num = np.bincount(true_labels.astype(int)[index==1]).argmax()
            label_reference[i] = num

        return label_reference
    
    
    def align_cluster_labels(self, cluster_labels, reference):
        ''' update the cluster labels to match the class labels'''
        aligned_lables = np.zeros_like(cluster_labels)
        for i in range(len(cluster_labels)):
            aligned_lables[i] = reference[cluster_labels[i]]

        return aligned_lables



##########################################################################
#--- Task 3 ---#
class MyLabelSelection:
    def __init__(self, ratio):
        self.ratio = ratio  # percentage of data to label
        ### TODO: Initialize other parameters needed in your algorithm

    def select(self, trainX):
        ''' Task 3-2'''
        

        # Return an index list that specifies which data points to label
        return data_to_label

    