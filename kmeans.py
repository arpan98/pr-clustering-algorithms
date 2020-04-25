from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix

from dataset import S_Sets, Dim_Sets, Spiral, Jain, Flame
from util import plot_silhouettes


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

def perform_k_means(n, data, true_labels=None, plot=False):
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(data)
    pred_labels = kmeans.predict(data)
    # prnt(pred_labels)
    # print(pred_labels[:200], true_labels[:200])

    silhouette_avg = silhouette_score(data, pred_labels)

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        plot_silhouettes(data, pred_labels, n, ax1)

    if true_labels:
        # print(true_labels[:10])
        # print(pred_labels[:10])
        # print(contingency_matrix(true_labels, pred_labels))
        label_map = np.argmax(contingency_matrix(true_labels, pred_labels), axis=1).tolist()
        # print("argmax ", np.argmax(contingency_matrix(true_labels, pred_labels), axis=1))

        def map_labels(x):
            try:
                return label_map.index(x) + 1
            except ValueError:
                return 0

        # print("label map ", label_map)
        mapped_pred_labels = list(map(map_labels, pred_labels))
        # print(mapped_pred_labels[:10])
        # com = [(true_labels[i], mapped_pred_labels[i]) for i in range(len(true_labels))]
        # print(com)
        print("The average silhouette_score is :", silhouette_avg)

        print("accuracy_score ", accuracy_score(true_labels, mapped_pred_labels))
        print("adjusted rand score ", adjusted_rand_score(true_labels, pred_labels))
        accuracy = calculate_accuracy(true_labels, pred_labels)
        print("my acc ", accuracy)
        # accuracy = accuracy_score()

    if plot:
        ax2.scatter(data['x'], data['y'], c=pred_labels, cmap='viridis')

        centers = kmeans.cluster_centers_
        # print(centers)
        ax2.scatter(centers[:, 0], centers[:, 1], c='black', s=50, alpha=0.5)

        plt.show()


def main():
    # for i in range(1, 5):
    data = Jain.get_data()
    # print(data)

    true_labels = Jain.get_labels()
    # print(data.describe())
    # plt.figure()
    # data.plot.scatter(x='x', y='y', marker='.', s=2)
    # plt.show()

    perform_k_means(2, data, true_labels, True)

if __name__ == "__main__":
    main()