import pandas as pd

class S_Sets:
    def __init__(self):
        self.data = [
            'data/s_sets/s1.txt',
            'data/s_sets/s2.txt',
            'data/s_sets/s3.txt',
            'data/s_sets/s4.txt'
        ]
        self.labels = [
            'data/s_sets/s1-label.pa',
            'data/s_sets/s2-label.pa',
            'data/s_sets/s3-label.pa',
            'data/s_sets/s4-label.pa'
        ]

    def get_data(self, i):
        data = pd.read_csv(self.data[i-1], sep="\s+", header=None)
        data.columns = ["x", "y"]
        return data

    def get_labels(self, i):
        read_labels = pd.read_csv(self.labels[i-1], header=None).to_numpy()
        true_labels = [a[0] for a in read_labels]
        return true_labels

class Dim_Sets:
    def __init__(self):
        self.data = [
            'data/s_sets/s1.txt',
            'data/s_sets/s2.txt',
            'data/s_sets/s3.txt',
            'data/s_sets/s4.txt'
        ]
        self.labels = [
            'data/s_sets/s1-label.pa',
            'data/s_sets/s2-label.pa',
            'data/s_sets/s3-label.pa',
            'data/s_sets/s4-label.pa'
        ]

    def get_data(self, i):
        data = pd.read_csv(self.data[i-1], sep="\s+", header=None)
        data.columns = ["x", "y"]
        return data

    def get_labels(self, i):
        read_labels = pd.read_csv(self.labels[i-1], header=None).to_numpy()
        true_labels = [a[0] for a in read_labels]
        return true_labels