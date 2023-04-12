from reduce_dimensions import *
from data_manip import *
from bayes import bayes_classifier
import matplotlib.pyplot as plt
from bayes import *

def classify_subject():
    def run_pca(train_data, train_labels, test_data, test_labels, d=0.95) -> float:
        # PCA
        train_data_pca, test_data_pca, d = apply_pca(train_data, test_data, d)
        separated_data = {}
        for i in train_labels:
            separated_data[i] = train_data_pca[i]
        pred_labels = []
        for i in test_data_pca:
            predicted = bayes_classifier(i, separated_data)
            pred_labels.append(predicted)
        acc = 0
        pred_labels.sort()
        test_labels.sort()
        for i, lab in enumerate(pred_labels):
            if lab == test_labels[i]:
                acc += 1
        accuracy = acc / len(test_labels)
        return accuracy

    train_data, train_labels, test_data, test_labels = load_data_pose()
    # train_data, train_labels, test_data, test_labels = load_data_illum()
    
    print('Running Bayes Subject Classification on POSE data')
    accuracy = run_pca(train_data, train_labels, test_data, test_labels)
    # run_mda(flattened, dataset, subjects, pics_each, d=5)
    # run_pca_plot(flattened)
    # run_mda_plot(flattened, dataset, subjects, pics_each)
    print(f'Accuracy using PCA: {round(accuracy*100, 2)}%')


def main():
    classify_subject()

if __name__ == '__main__':
    main()