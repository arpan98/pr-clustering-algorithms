from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, adjusted_rand_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix
from dataset import S_Sets


def calculate_accuracy(y_true, y_pred):
    label_map = {}
    true_labels_distinct = [i for i in range(1, 16)]
    # true_labels_distinct = sorted(list(set(y_true)))
    l_true = 1
    start = 0
    end = 0
    for i in range(len(y_true)):
        if y_true[i] != l_true or i == len(y_true)-1:
            end = i
            c = Counter(y_pred[start:end+1])
            # if c:
            l_p = c.most_common(1)[0][0]
            label_map[l_true] = l_p
            l_true += 1
            start = end + 1
    num_correct = Counter([1 if label_map[y_true[i]] == y_pred[i] else 0 for i in range(len(y_pred))])[1]
    accuracy = num_correct * 100 / len(y_pred)
    return accuracy


def perform_hierarchical_clustering(n, data, true_labels=None):
    model = GaussianMixture(n_components=15, covariance_type='full', n_init=10)
    pred_labels = model.fit_predict(data)
    if true_labels:
        label_map = np.argmax(contingency_matrix(true_labels, pred_labels), axis=1).tolist()
        print(contingency_matrix(true_labels, pred_labels))
        print("argmax ", np.argmax(contingency_matrix(true_labels, pred_labels), axis=1))

        def map_labels(x):
            try:
                return label_map.index(x) + 1
            except ValueError:
                return 0

        # print("label map ", label_map)
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("The average silhouette_score is :", silhouette_avg)
        mapped_pred_labels = list(map(map_labels, pred_labels))
        print("accuracy_score ", accuracy_score(true_labels, mapped_pred_labels))
        print("adjusted rand score ", adjusted_rand_score(true_labels, pred_labels))
        accuracy = calculate_accuracy(true_labels, pred_labels)
        print("my acc ", accuracy)
        # accuracy = accuracy_score()

    plt.scatter(data['x'], data['y'], c=pred_labels, cmap='viridis')

    plt.show()    


def main():
    for i in range(1, 5):
        s_sets = S_Sets()
        data = s_sets.get_data(i)
        # print(data)

        true_labels = s_sets.get_labels(i)
        # print(data.describe())
        # plt.figure()
        # data.plot.scatter(x='x', y='y', marker='.', s=2)
        # plt.show()

        perform_hierarchical_clustering(15, data, true_labels)

if __name__ == "__main__":
    main()