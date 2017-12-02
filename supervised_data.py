import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from abc import abstractmethod


class SupervisedData:
    def __init__(self):
        # Load data
        self.x, self.y = self.load()
        self.number_of_samples = len(self.x)
        self.number_of_features = len(self.x[0])

        # Create labels
        self.feature_labels = self.label_features()
        self.class_labels = self.label_classes()
        self.number_of_classes = len(self.class_labels)

        # Store as dataframe
        self.dataframe = self.create_dataframe()

        # Initialize data separations
        self.training_x = None
        self.training_y = None
        self.testing_x = None
        self.testing_y = None

        self.k_fold_splits = None  # list of cross validation indices for training_x and training_y

    # Loading and Labeling
    @abstractmethod
    def load(self):
        """
        :return: x, y
        x is an mxn numpy array of the data where m is the number of samples and n is the number of features.
        y is an mx1 numpy array of the classes where m is the number of samples.
        Each value in y is an integer class label indexed from 0.
        """
        raise NotImplementedError

    @abstractmethod
    def label_features(self):
        """
        :return: feature_labels
        feature_labels is an n length list of strings of the label for each feature, where n is the number of features.
        """
        raise NotImplementedError

    @abstractmethod
    def label_classes(self):
        """
        :return: class_labels
        class_labels is an l length list of strings of the label for each class, where l is the number of classes.
        """
        raise NotImplementedError

    def create_dataframe(self):
        """
        :return: dataframe
        dataframe is a pandas dataframe containing the labeled data and class information
        """
        dataframe = pd.DataFrame(data=np.concatenate((self.x, self.y), axis=1),
                                 columns=(self.feature_labels + ['Class']))
        return dataframe

    # Statistics
    def minimums(self):
        return self.dataframe.min()

    def maximums(self):
        return self.dataframe.max()

    def modes(self):
        return self.dataframe.mode()

    def means(self):
        return self.dataframe.mean()

    def standard_deviations(self):
        return self.dataframe.std()

    # Data operation prep
    def train_test_split(self, test_size, shuffle=True, random_state=None):
        """
        :param test_size: fraction of the data that should be held aside for testing.
        :param shuffle: whether or not the input data should be shuffled.
        :param random_state: seed for random shuffling.

        produces x and y data split into training and testing sets
        """
        self.training_x, self.testing_x, self.training_y, self.testing_y = \
            train_test_split(self.x, self.y, test_size=test_size, shuffle=shuffle, random_state=random_state)

    def k_fold(self, k, shuffle=False, random_state=None):
        """
        :param k: number of folds for cross validation
        :param shuffle: whether or not the input data should be shuffled.
        :param random_state: seed for random shuffling.

        produces k_fold indices for cross validation
        """
        # The result can be used as follows
        # for training_index, validation_index in self.k_fold_splits:
        #     training_x = self.training_x[training_index]
        #     training_y = self.training_y[training_index]
        #     validation_x = self.training_x[validation_index]
        #     validation_y = self.training_y[validation_index]

        if self.training_x is None or self.training_y is None:
            print("Warning: 20% of the data has be held aside for testing in "
                  "self.testing_x, self.testing_y by default.\n"
                  "If you would like a custom amount, use self.train_test_split() first.")
            self.train_test_split(0.2)

        kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
        self.k_fold_splits = list(kf.split(self.training_x))

    # Data reduction
    def pca_reduce(self, number_of_components):
        """
        :param number_of_components: number of components to reduce to
        :return: principal_components
        principal_components is an mxn numpy array of the principal components m is the number of samples and n is the
        number of principal components
        """

        if number_of_components > self.number_of_features:
            print("Error: The number of features is {} and cannot be reduced to {}.".format(
                self.number_of_features, number_of_components))
            exit(1)

        pca = PCA(n_components=number_of_components)
        pca.fit(self.x)
        principal_components = pca.transform(self.x)
        return principal_components

    def lda_reduce(self, number_of_components):
        """
        :param number_of_components: number of components to reduce to
        :return: principal_components
        principal_components is an mxn numpy array of the principal components m is the number of samples and n is the
        number of principal components
        """

        if number_of_components >= self.number_of_classes:
            print("Error: The number of classes is {}. The number of components must be less instead of {}.".format(
                self.number_of_classes, number_of_components))
            exit(1)

        lda = LinearDiscriminantAnalysis(n_components=number_of_components)
        principal_components = lda.fit_transform(self.x, self.y.ravel())
        return principal_components

    # Graphing
    def pca_graph(self, to_file=False, filename=None):
        """
        Plot the data in two dimensions by reducing using principal component analysis
        """

        principal_components = self.pca_reduce(2)

        for c in range(self.number_of_classes):
            indices = np.where(self.y == c)[0]  # get all indices for this class
            class_principal_components = principal_components[indices]
            principal_component_0 = class_principal_components[:, 0]
            principal_component_1 = class_principal_components[:, 1]
            plt.scatter(principal_component_0, principal_component_1, s=1, label=self.class_labels[c])

        plt.legend()

        if to_file:
            if filename is None:
                print("Error: No filename provided for saved graph.")
                exit(1)
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def lda_graph(self, to_file=False, filename=None):
        """
        Plot the data in two dimensions by reducing using principal component analysis
        """

        principal_components = self.lda_reduce(1)
        for c in range(self.number_of_classes):
            indices = np.where(self.y == c)[0]  # get all indices for this class
            class_principal_components = principal_components[indices]
            principal_component_0 = class_principal_components[:, 0]
            plt.scatter(principal_component_0, [c] * len(principal_component_0), s=1, label=self.class_labels[c])

        plt.legend()

        if to_file:
            if filename is None:
                print("Error: No filename provided for saved graph.")
                exit(1)
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def boxplot(self, to_file=False, filename=None):
        """
        Plot the data as a boxplot with no outlier points
        """
        plt.boxplot(self.x, 0, '')

        if to_file:
            if filename is None:
                print("Error: No filename provided for saved graph.")
                exit(1)
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
