import math
import json
import random
from neural_node import NeuralNode

THRESHOLD_PERCENTAGE = 1.0
THRESHOLD_MIN_ENTRIES = 50
FOREST_SIZE = 50

################################################
# CIS 678 Decision Tree Learning
#
# parses data, creates neural network, checks results
################################################
def decision_forest():
    data = None
    with open("agaricus-lepiota.data") as f:
        data = [line.split(',') for line in f.read().splitlines()]
    for entry in data:
        del entry[11]

    random.shuffle(data)

    training_sets = []
    training_size = 6500 // FOREST_SIZE
    for i in range(0, FOREST_SIZE):
        training_sets.append(data[i * training_size:i * training_size + training_size])
    testing_set = data[6500:]
    
    training_classification_sets = []
    for training_set in training_sets:
        training_classification_sets.append([line.pop(0) for line in training_set])
    testing_classifications = [line.pop(0) for line in testing_set]

    attributes = []
    for training_set in training_sets:
        attributes.append([])
        for i in range(0, len(training_set[0])):
            attributes[-1].append([line[i] for line in training_set])

    forest = []
    for i in range(0, FOREST_SIZE):
        forest.append(NeuralNode())
        expand_network(forest[i], training_sets[i], training_classification_sets[i], attributes[i])

    #validate
    for i in range(0, len(forest)):
        print("Tree %d" % i)
        print("--------------------")
        forest[i].print()
    
    num_correct = 0
    for i in range(0, len(testing_set)):
        if calculate_decision(forest, testing_set[i]) == testing_classifications[i]:
            num_correct += 1

    print("%d%%" % (num_correct / len(testing_classifications) * 100))

################################################
# calculates decision of forest
#
# @param forest current forest function is working on
# @param entry data entry decision is being made on
#
# @return char 'e' for edible, 'p' for poisonous
################################################
def calculate_decision(forest, entry):
    num_e = 0
    num_p = 0
    for tree in forest:
        if calculate_decision_tree(tree, entry) == 'e':
            num_e += 1
        else:
            num_p += 1
    if num_e > num_p:
        return 'e'
    else:
        return 'p'

################################################
# calculates decision of tree
#
# @param node current node function is working on
# @param entry data entry decision is being made on
#
# @return char 'e' for edible, 'p' for poisonous
################################################
def calculate_decision_tree(node, entry):
    if node.leaf == True:
        return node.decision
    else:
        no_matches = True
        for child in node.children:
            if entry[child.attribute_index] == child.attribute_value:
                no_matches = False
                return calculate_decision_tree(child, entry)

################################################
# recursively creates neural network
#
# @param node current node
# @param data current set of data
# @param classifications current classifications for data
# @param attributes current attributes for data
################################################
def expand_network(node, data, classifications, attributes):
    p = sum([1 for c in classifications if c == 'e']) / len(classifications)
    if p >= THRESHOLD_PERCENTAGE:
        node.leaf = True
        node.decision = 'e'
        return
    elif 1.0 - p >= THRESHOLD_PERCENTAGE:
        node.leaf = True
        node.decision = 'p'
        return

    if len(attributes) == 0:
        if p >= 1 - p:
            node.leaf = True
            node.decision = 'e'
            return
        else:
            node.leaf = True
            node.decision = 'p'
            return

    #calc entropy of system
    entropy_system = p_log_p(p)

    gain_attributes = []
    for a in attributes:
        if a == "used":
            gain_attributes.append(-1)
            continue
        a_count = {}
        for i in range(0, len(a)):
            if a[i] not in a_count.keys():
                a_count[a[i]] = []
            a_count[a[i]].append(i)
        entropy_attribute = 0
        for a_type in a_count:
            a_type_edible = sum([1 for index in a_count[a_type] if classifications[index] == 'e'])
            edible_ratio = a_type_edible / len(a_count[a_type])
            entropy_attribute += len(a_count[a_type]) / len(a) * p_log_p(edible_ratio)
        gain_attributes.append(entropy_system - entropy_attribute)

    selection = gain_attributes.index(max(gain_attributes))
    a_count = {}
    for i in range(0, len(attributes[selection])):
        if attributes[selection][i] not in a_count.keys():
            a_count[attributes[selection][i]] = []
        a_count[attributes[selection][i]].append(i)
    for a_type in a_count:
        n = NeuralNode()
        n.attribute_index = selection
        n.attribute_value = a_type
        node.children.append(n)

        new_attributes = attributes[:]
        new_attributes[selection] = "used"
        for i in range(0, len(new_attributes)):
            if new_attributes[i] == "used":
                continue
            new_attributes[i] = [new_attributes[i][c] for c in range(0, len(new_attributes[i])) if c in a_count[a_type]]
        expand_network(node.children[-1], [data[i] for i in a_count[a_type]], [classifications[i] for i in a_count[a_type]], new_attributes)

################################################
# computes entropy
#
# @param ratio the ratio used to calc entropy
#
# @return double the calculated entropy
################################################
def p_log_p(ratio):
    if (ratio == 0.0 or ratio == 1.0):
        return 0.0
    return - ratio * math.log2(ratio) - (1 - ratio) * math.log2(1 - ratio)

################################################
# starts program by running main function
################################################
if __name__ == "__main__":
    decision_forest()