from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def apply_pca(train_data, test_data, dimensions=0.95):
    pca = PCA(n_components=dimensions)
    pca = pca.fit(train_data) # fit model to training data
    train_data_pca = pca.transform(train_data) # transform training data
    test_data_pca = pca.transform(test_data) # transform testing data
    return train_data_pca, test_data_pca, pca.n_components_

def apply_mda(train_data, train_labels, test_data, d=20):
    mda = LDA(n_components=d)
    mda.fit(train_data, train_labels)
    train_data_mda = mda.transform(train_data)
    test_data_mda = mda.transform(test_data)
    return train_data_mda, test_data_mda