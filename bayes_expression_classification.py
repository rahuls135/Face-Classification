from reduce_dimensions import *
from data_manip import *
from bayes import bayes_classifier
import matplotlib.pyplot as plt
from bayes import *

def classify_subject():
    def reduce_and_run(train_data, train_labels, test_data, test_labels, d=0.95) -> float:
        train_data_pca, test_data_pca, pca_d = apply_pca(train_data, test_data, d)
        d = 1
        train_data_mda, test_data_mda = apply_mda(train_data_pca, train_labels, test_data_pca, d=d)
        separated_data = {}
        for i in train_labels:
            separated_data[i] = train_data_mda[i]
        pred_labels = []
        for i in test_data_mda:
            predicted = bayes_classifier(i, separated_data)
            pred_labels.append(predicted)
        acc = 0
        pred_labels.sort()
        test_labels.sort()
        for i, lab in enumerate(pred_labels):
            if lab == test_labels[i]:
                acc += 1
        accuracy = acc / len(test_labels)
        return accuracy, pca_d

    train_data, train_labels, test_data, test_labels = load_data_face()
    # train_data, train_labels, test_data, test_labels = load_data_illum()
    
    print('Running Bayes Expression Classification on FACE data')
    listd = list(range(50,225,25))
    accuracies = []
    for d in listd:
        accuracy, pca_d= reduce_and_run(train_data, train_labels, test_data, test_labels, d)
        accuracies.append(accuracy)
        print(f'Accuracy using PCA and MDA with {pca_d} dimensions: {round(accuracy*100, 2)}%')

    X = listd
    Y = accuracies
    plt.plot(X, Y)
    plt.xlabel("PCA Dimensions")
    plt.ylabel("Accuracy (%)")
    plt.title("Bayes Expression Classification")
    plt.savefig('bayes_expr_class')
    plt.show()


def main():
    classify_subject()

if __name__ == '__main__':
    main()