from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5: 0.45,
                 0.25: 1.32,
                 0.1: 2.71,
                 0.05: 3.84,
                 0.0001: 100000},
             2: {0.5: 1.39,
                 0.25: 2.77,
                 0.1: 4.60,
                 0.05: 5.99,
                 0.0001: 100000},
             3: {0.5: 2.37,
                 0.25: 4.11,
                 0.1: 6.25,
                 0.05: 7.82,
                 0.0001: 100000},
             4: {0.5: 3.36,
                 0.25: 5.38,
                 0.1: 7.78,
                 0.05: 9.49,
                 0.0001: 100000},
             5: {0.5: 4.35,
                 0.25: 6.63,
                 0.1: 9.24,
                 0.05: 11.07,
                 0.0001: 100000},
             6: {0.5: 5.35,
                 0.25: 7.84,
                 0.1: 10.64,
                 0.05: 12.59,
                 0.0001: 100000},
             7: {0.5: 6.35,
                 0.25: 9.04,
                 0.1: 12.01,
                 0.05: 14.07,
                 0.0001: 100000},
             8: {0.5: 7.34,
                 0.25: 10.22,
                 0.1: 13.36,
                 0.05: 15.51,
                 0.0001: 100000},
             9: {0.5: 8.34,
                 0.25: 11.39,
                 0.1: 14.68,
                 0.05: 16.92,
                 0.0001: 100000},
             10: {0.5: 9.34,
                  0.25: 12.55,
                  0.1: 15.99,
                  0.05: 18.31,
                  0.0001: 100000},
             11: {0.5: 10.34,
                  0.25: 13.7,
                  0.1: 17.27,
                  0.05: 19.68,
                  0.0001: 100000}}


def get_column(data, column_index):
    """
    Get the 'i' column from the dataset.
 
    Input:
    - data: any dataset.
 
    Returns:
    - column: The 'i' column.
    """
    ###########################################################################
    # Implemention of the function.                                           #
    ###########################################################################
    column = data[:, column_index]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return column


def feature_attributes_and_occurrences(data, feature=-1):
    """
    Get the occurrences of each of the attribute of the feature from the dataset column.
 
    Input:
    - data: Any dataset.
    - feature: The column (feature) index to work on.
 
    Returns:
    - value_occur_dict: A dict with a attribite value and his occurences.
    """
    ###########################################################################
    # Implemention of the function.                                           #
    ###########################################################################
    class_column = get_column(data, feature)
    attributes = np.unique(class_column)
    value_occur_dict = {attribute: [index for index, value in enumerate(class_column) if value == attribute]
                        for attribute in attributes}
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return value_occur_dict


def filter_data_for_features_attribute(data, attribute_value, attribute_index):
    """
    Count the occurrences of each of the attribute values from the dataset column.
 
    Input:
    - data: Any dataset.
    - attribute_value: The attribute value to filter by.
    - attribute_index: The attribute value to filter by.
 
    Returns:
    - filter_data: A new data containing only the filtered rows.
    """
    ###########################################################################
    # Implemention of the function.                                           #
    ###########################################################################

    counter = feature_attributes_and_occurrences(data, attribute_index)

    filter_data = data[counter.get(attribute_value), :]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return filter_data


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # Implemention of the function.                                           #
    ###########################################################################
    num_of_instances = data.shape[0]

    gini = 1

    counter = feature_attributes_and_occurrences(data)

    for attribute_value in counter:
        gini -= (len(counter.get(attribute_value)) / num_of_instances) ** 2
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # Implemention of the function.                                           #
    ###########################################################################
    num_of_instances = data.shape[0]

    counter = feature_attributes_and_occurrences(data)

    for attribute_value in counter:
        attribute_part = (len(counter.get(attribute_value)) / num_of_instances)
        entropy -= attribute_part * np.log2(attribute_part)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy


def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """
    goodness = 0
    groups = {}  # groups[feature_value] = data_subset
    ###########################################################################
    # Implement the function.                                                 #
    ###########################################################################

    if gain_ratio:
        impurity_func = calc_entropy

    goodness = impurity_func(data)
    counter = feature_attributes_and_occurrences(data, feature)

    split_in_info = 0

    for attribute in counter:
        filtered_data = filter_data_for_features_attribute(data, attribute, feature)
        groups[attribute] = filtered_data
        goodness -= len(filtered_data) / data.shape[0] * impurity_func(filtered_data)

        # Some calculations in case that gain_ratio flag is on.
        part_of_data = len(filtered_data) / data.shape[0]
        split_in_info -= part_of_data * np.log2(part_of_data)

    if gain_ratio and split_in_info != 0:
        goodness = goodness / split_in_info

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return goodness, groups


class DecisionNode:

    def __init__(self, data, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data  # the relevant data for the node
        self.feature = feature  # column index of criteria being tested
        self.pred = self.calc_node_pred()  # the prediction of the node
        self.depth = depth  # the current depth of the node
        self.children = []  # array that holds this nodes children
        self.children_values = []
        self.terminal = False  # determines if the node is a leaf
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio

    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################

        labels_count = Counter(self.data[:, -1])

        pred = max(labels_count, key=labels_count.get)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)

    def split(self, impurity_func):
        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        ###########################################################################
        # Implement the function.                                                 #
        ###########################################################################

        # Checks if we reach the max depth.
        if self.depth == self.max_depth:
            self.terminal = True
            return

        num_of_features = self.data.shape[1] - 1
        # Dict holder for best feature split
        best_groups = {}
        best_feature_i = 0
        best_feature_goodness = -1

        # Finding best feature and his group.
        for feature_i in range(0, num_of_features):
            if feature_i != self.feature:
                # Current feature split and impurity
                feature_i_goodness, groups_i = goodness_of_split(self.data, feature_i, impurity_func, self.gain_ratio)
                if feature_i_goodness > best_feature_goodness:
                    best_groups = groups_i
                    best_feature_i = feature_i
                    best_feature_goodness = feature_i_goodness

        # Check the chi
        cur_chi, deg_of_freedom = chi_compute(self);
        if self.chi < 1 and cur_chi > chi_table.get(deg_of_freedom).get(self.chi):
            self.terminal = True
            return

        self.feature = best_feature_i

        # Adding children nodes according to best attribute
        for classifier, sub_data in best_groups.items():
            child_node = DecisionNode(data=sub_data, feature=self.feature, depth=self.depth + 1, chi=self.chi,
                                      max_depth=self.max_depth,
                                      gain_ratio=self.gain_ratio)
            self.add_child(child_node, classifier)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset.
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """
    root = DecisionNode(data, feature=-1, depth=0, chi=chi, max_depth=max_depth, gain_ratio=gain_ratio)
    ###########################################################################
    # Implement the function.                                                 #
    ###########################################################################

    build_sub_tree(root, impurity)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return root


def build_sub_tree(node, impurity):
    """
    Build a tree using the given impurity measure and training dataset where
    the root is the given node.

    Input:
    - node: the root.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: This function has no return value.
    """
    ###########################################################################
    # Implement the function.                                                 #
    ###########################################################################
    if impurity(node.data) != 0:
        node.split(impurity)
        if not node.terminal:
            for child_node in node.children:
                build_sub_tree(child_node, impurity)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def predict(root, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    pred = None
    ###########################################################################
    # Implement the function.                                                 #
    ###########################################################################

    temp_node = root

    while not temp_node.terminal:

        # Finding the correct node for the instance
        if instance[temp_node.feature] in temp_node.children_values:
            temp_node = temp_node.children[temp_node.children_values.index(instance[temp_node.feature])]
        else:
            break

    pred = temp_node.pred

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pred


def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    count_correct = 0
    ###########################################################################
    # Implement the function.                                                 #
    ###########################################################################

    for instance in dataset:
        if predict(node, instance) == instance[-1]:
            count_correct += 1

    accuracy = count_correct / dataset.shape[0] * 100

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy


def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels

    Output: the training and testing accuracies per max depth
    """
    training = []
    testing = []
    ###########################################################################
    # Implement the function.                                                 #
    ###########################################################################

    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = build_tree(X_train, impurity=calc_entropy, gain_ratio=True, max_depth=max_depth)
        testing.append(calc_accuracy(tree, X_test))
        training.append(calc_accuracy(tree, X_train))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return training, testing


def chi_compute(node):
    """
        Calculate chi of given node.
    """
    chi_square = 0
    ###########################################################################
    # Implement the function.                                                 #
    ###########################################################################
    label_occur = feature_attributes_and_occurrences(node.data, -1)
    labels_prob = {label: (len(occur) / node.data.shape[0]) for label, occur in label_occur.items()}
    feature_label = feature_attributes_and_occurrences(node.data, node.feature)
    deg_of_freedom = len(feature_label.keys())
    for feature_value, value_count in feature_label.items():
        for class_value, class_probability in labels_prob.items():
            expected = len(value_count) * class_probability
            observed = np.sum((node.data[:, node.feature] == feature_value) & (node.data[:, -1] == class_value))
            chi_square += (observed - expected) ** 2 / expected
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return chi_square, deg_of_freedom


def chi_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels

    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc = []
    depth = []
    ###########################################################################
    # Implement the function.                                                 #
    ###########################################################################
    for chi_i in [1 , 0.5, 0.25, 0.1, 0.05, 0.0001]:
        # entropy and gain ratio
        chi_i_tree = build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True, chi=chi_i)
        chi_training_acc.append(calc_accuracy(chi_i_tree, X_train))
        chi_testing_acc.append(calc_accuracy(chi_i_tree, X_test))

        depth.append(tree_depth(chi_i_tree))
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return chi_training_acc, chi_testing_acc, depth

def tree_depth(node):
    """
        Calculate tree depth (recur).
    """
    ###########################################################################
    # Implement the function.                                                 #
    ###########################################################################
    if node.terminal:
        return 1
    
    depth = 0

    for child in node.children:
        child_depth = tree_depth(child)
        if child_depth > depth:
            depth = child_depth
    
    depth += 1

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return depth

def count_nodes(node):
    """
    Count the number of node in a given tree

    Input:
    - node: a node in the decision tree.

    Output: the number of nodes in the tree.
    """
    n_nodes = None
    ###########################################################################
    # Implement the function.                                                 #
    ###########################################################################

    if not node.children:
        return 1

    n_nodes = 1

    for child in node.children:
        n_nodes += count_nodes(child)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes
