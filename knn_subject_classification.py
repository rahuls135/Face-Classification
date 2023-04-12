from reduce_dimensions import *
from data_manip import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as MDA
from knn_classifier import run_knn
import scipy.interpolate as interp

def knn_subject_classification():
    train_data, train_labels, test_data, test_labels = load_data_pose()

    # Perform PCA transformation
    train_data_pca, test_data_pca, d = apply_pca(train_data, test_data)

    # Fit MDA to training data
    d = 20
    k = 3
    accuracies = []
    listd = list(range(10,70,10))
    listk = list(range(1,7))

    for d in listd:
        for k in listk:
            train_data_mda, test_data_mda = apply_mda(train_data, train_labels, test_data, d=d)

        # Apply k-NN classifier to test data
            predicted_labels = run_knn(train_data_mda, test_data_mda, train_labels, k=k)

            accuracy = get_accuracy(predicted_labels, test_labels)
            accuracy = round(accuracy*100, 2)
            print(f"Accuracy with {d} MDA dimensions and {k} neighbors: {accuracy}%")
            accuracies.append(accuracy)

    # List comprehension to make 3 equal length lists for graphing purposes
    oldlen = len(listd)
    listk *= oldlen
    newd = [elem for elem in listd for _ in range(oldlen)]
    listd.extend(newd)
    listd = listd[oldlen:]

    
    X = listd
    Y = listk
    Z = accuracies

    plotx,ploty, = np.meshgrid(np.linspace(np.min(X),np.max(X),10),\
                            np.linspace(np.min(Y),np.max(Y),10))
    plotz = interp.griddata((X,Y),Z,(plotx,ploty),method='linear')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(plotx,ploty,plotz,cstride=1,rstride=1,cmap='viridis')
    
    # Change labels and title
    ax.set_xlabel('MDA Dimensions')
    ax.set_ylabel('Nearest Neighbors')
    ax.set_zlabel('Accuracy (%)')
    ax.set_title('K-NN Subject Classification only MDA')
    ax.view_init(elev=25, azim=45) # elevation angle and rotational angle
    fig.savefig('knn_subject_classification_mda.png', dpi=300)
    plt.show()

def get_accuracy(predicted_labels, test_labels):
    correct = 0
    for i in range(len(test_labels)):
        if predicted_labels[i] == test_labels[i]:
            correct += 1
    accuracy = correct / len(test_labels)
    return accuracy

def main():
    knn_subject_classification()

if __name__ == '__main__':
    main()
    