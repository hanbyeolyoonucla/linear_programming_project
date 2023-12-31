import numpy as np
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
### TODO: import any other packages you need for your solution

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

        # initial centroids: randomly select K points from trainX
        self.cluster_centers_ = trainX[np.random.choice(trainX.shape[0], size=self.K, replace=False)]

        # while labels changes
        itr = 0
        max_itr = 10000
        while itr < max_itr:
            itr = itr + 1

            # distance of points from centroids n x K
            vector_from_centers = trainX.reshape(trainX.shape[0], 1, trainX.shape[1]) - self.cluster_centers_  # n x K x m
            distance_from_centers = np.linalg.norm(vector_from_centers, axis=2)  # n x K

            # update labels
            new_labels = np.argmin(distance_from_centers, axis=1)
            if np.all(new_labels == self.labels):
                break
            else:
                self.labels = new_labels

            # update cluster centroids
            for i in np.arange(self.K):
                self.cluster_centers_[i, :] = np.mean(trainX[self.labels == i], axis=0)

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

    