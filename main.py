from utils import *
from MySolution_LP import MyClassifier, MyClustering, MyLabelSelection


if __name__ == '__main__':

    # Get synthetic datasets
    syn_data = prepare_synthetic_data()
    print("Synthetic data shape: ", syn_data['trainX'].shape, syn_data['trainY'].shape)
    plt.scatter(syn_data['trainX'][:, 0], syn_data['trainX'][:, 1], c=syn_data['trainY'])
    plt.show()

    # Get mnist datasets
    mnist_data = prepare_mnist_data()
    print("MNIST data shape: ", mnist_data['trainX'].shape, mnist_data['trainY'].shape)
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(mnist_data['trainX'][i].reshape(28, 28), cmap='gray')
    plt.show()

    # Train clustering
    clustering_syn_3 = MyClustering(3)
    clustering_syn_10 = MyClustering(10)
    clustering_syn_32 = MyClustering(32)
    clustering_mnist_3 = MyClustering(3)
    clustering_mnist_10 = MyClustering(10)
    clustering_mnist_32 = MyClustering(32)
    clustering_syn_3.train(syn_data['trainX'])
    clustering_syn_10.train(syn_data['trainX'])
    clustering_syn_32.train(syn_data['trainX'])
    clustering_mnist_3.train(mnist_data['trainX'])
    clustering_mnist_10.train(mnist_data['trainX'])
    clustering_mnist_32.train(mnist_data['trainX'])

    # Evaluate clustering performance
    print(clustering_syn_3.evaluate_clustering(syn_data['trainY']))
    print(clustering_syn_10.evaluate_clustering(syn_data['trainY']))
    print(clustering_syn_32.evaluate_clustering(syn_data['trainY']))
    print(clustering_mnist_3.evaluate_clustering(mnist_data['trainY']))
    print(clustering_mnist_10.evaluate_clustering(mnist_data['trainY']))
    print(clustering_mnist_32.evaluate_clustering(mnist_data['trainY']))

    #
    print(clustering_syn_3.evaluate_classification(syn_data['trainY'], syn_data['testX'], syn_data['testY']))
    print(clustering_syn_10.evaluate_classification(syn_data['trainY'], syn_data['testX'], syn_data['testY']))
    print(clustering_syn_32.evaluate_classification(syn_data['trainY'], syn_data['testX'], syn_data['testY']))
    print(clustering_mnist_3.evaluate_classification(mnist_data['trainY'], mnist_data['testX'], mnist_data['testY']))
    print(clustering_mnist_10.evaluate_classification(mnist_data['trainY'], mnist_data['testX'], mnist_data['testY']))
    print(clustering_mnist_32.evaluate_classification(mnist_data['trainY'], mnist_data['testX'], mnist_data['testY']))

    #
    # plt.scatter(syn_data['trainX'][:, 0], syn_data['trainX'][:, 1], c=clustering.labels)
    # plt.scatter(clustering.cluster_centers_[:, 0], clustering.cluster_centers_[:, 1], color='red', marker='x')
    # # plt.show()
    #
    # # Test clustering
    # pred_labels = clustering.infer_cluster(syn_data['testX'])
    # plt.scatter(syn_data['testX'][:, 0], syn_data['testX'][:, 1], c=pred_labels)
    # plt.scatter(clustering.cluster_centers_[:, 0], clustering.cluster_centers_[:, 1], color='red', marker='x')
    # # plt.show()



