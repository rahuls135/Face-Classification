from reduce_dimensions import *
from data_manip import *
from knn_classifier import run_knn

def knn_expression_classification():
    train_data, train_labels, test_data, test_labels = load_data_face()

    listk = list(range(1,17, 2))

    # K-NN Only PCA
    accuracies = []
    train_data_pca, test_data_pca, d = apply_pca(train_data, test_data)

    for k in listk:
        predicted_labels = run_knn(train_data_pca, test_data_pca, train_labels, k=k)
        accuracy = get_accuracy(predicted_labels, test_labels)
        accuracy = round(accuracy*100, 2)
        accuracies.append(accuracy)
        print(f"Accuracy with only PCA and {k} neighbors: {accuracy}")
    X = listk
    Y = accuracies
    plt.plot(X, Y)
    plt.xlabel("Nearest Neighbors")
    plt.ylabel("Accuracy (%)")
    plt.title("K-NN Expression Classification only PCA")
    plt.savefig('knn_expr_class_pca')
    plt.show()

    # K-NN Only MDA
    accuracies = []
    d = 1 # numclasses = 2
    train_data_mda, test_data_mda = apply_mda(train_data, train_labels, test_data, d=d)
    for k in listk:
        predicted_labels = run_knn(train_data_mda, test_data_mda, train_labels, k=k)
        accuracy = get_accuracy(predicted_labels, test_labels)
        accuracy = round(accuracy*100, 2)
        accuracies.append(accuracy)
        print(f"Accuracy with only MDA and {k} neighbors: {accuracy}")
    X = listk
    Y = accuracies
    plt.plot(X, Y)
    plt.xlabel("Nearest Neighbors")
    plt.ylabel("Accuracy (%)")
    plt.title("K-NN Expression Classification only MDA")
    plt.savefig('knn_expr_class_mda')
    plt.show()

    # K-NN PCA and MDA
    accuracies = []
    train_data_pca, test_data_pca, d = apply_pca(train_data, test_data)
    d = 1 # numclasses = 2
    train_data_mda, test_data_mda = apply_mda(train_data_pca, train_labels, test_data_pca, d=d)
    for k in listk:
        predicted_labels = run_knn(train_data_mda, test_data_mda, train_labels, k=k)
        accuracy = get_accuracy(predicted_labels, test_labels)
        accuracy = round(accuracy*100, 2)
        accuracies.append(accuracy)
        print(f"Accuracy with {k} neighbors: {accuracy}")
    X = listk
    Y = accuracies
    plt.plot(X, Y)
    plt.xlabel("Nearest Neighbors")
    plt.ylabel("Accuracy (%)")
    plt.title("K-NN Expression Classification")
    plt.savefig('knn_expr_class')
    plt.show()


def get_accuracy(predicted_labels, test_labels):
    correct = 0
    for i in range(len(test_labels)):
        if predicted_labels[i] == test_labels[i]:
            correct += 1
    accuracy = correct / len(test_labels)
    return accuracy

def main():
    knn_expression_classification()

if __name__ == '__main__':
    main()
    