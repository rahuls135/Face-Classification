import numpy as np


def run_knn(train_data, test_data, train_labels, k=3):
    num_classes = len(train_labels)
    predicted_labels = []
    for i in range(len(test_data)):
        distances = []
        for j in range(len(train_data)):
            distance = np.linalg.norm(test_data[i] - train_data[j])
            distances.append((distance, train_labels[j]))
        distances.sort()
        neighbors = distances[:k]
        counts = np.zeros(num_classes)
        for neighbor in neighbors:
            counts[int(neighbor[1])] += 1
        predicted_label = np.argmax(counts)
        predicted_labels.append(predicted_label)
    return predicted_labels