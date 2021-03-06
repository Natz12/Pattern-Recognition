

# Import the libraries
import pandas as pd
import numpy as np
import seaborn as sns
# import math

import statistics # mode() function a sub-set of the statistics module

# Try getting the file online if not found try it locally
try:
  print("Getting wine dataset file from the internet")
  df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
               header = None) #Load from url
except:
  print("Could not find the wine dataset, getting the file locally")
  df_wine = pd.read_csv('wine.data',
               header=None) 
else:
  print("Wine dataset file loaded from the web")


np_wine = df_wine.values

# Try getting the file online if not found try it locally
try:
  print("Getting wine dataset file from the internet")
  df_ttt = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data',
               header = None) #Load from url
except:
  print("Could not find the wine dataset, getting the file locally")
  df_ttt = pd.read_csv('tic-tac-toe.data',
               header=None) 
else:
  print("Wine dataset file loaded from the web")

np_ttt = df_ttt.values
# df_wine[50:100]
# array_wine
# df_ttt[0].unique()
df_wine.rename(columns = {0:'truth'})
df_wine

class Tree:
  # def __init__(self, feature, threshold, leaf, prediction, left, right):
  #   self.feature = feature
  #   self.threshold = None
  #   self.leaf = True
  #   self.prediction = False
  #   self.left = None
  #   self.right = None
  leaf = True
  prediction = False
  feature = None
  threshold = None
  left = None
  right = None

# accepts numpy array
def train(data,  class_column = 0, features_ignore = [], use_IG = True):

  """Return the trained ID3 tree.

  Trains a decision tree using ID3 algorithm. 

  Parameters: 
  # X (np.array): np.array with the features of the samples in the training set
  # y (np.array): np.array with the labels of the samples in the training set
  data: np.array with the samples in the training set
  class_column: index of the column having the class

  Returns: 
  object of class Tree.


  #https://docs.python.org/3/library/doctest.html
  # >>> [factorial(n) for n in range(6)]
  # [1, 1, 2, 6, 24, 120]
  # >>> factorial(30)
  # 265252859812191058636308480000000
  # >>> factorial(-1)
  # Traceback (most recent call last):
  #     ...
  # ValueError: n must be >= 0

  # Factorials of floats are OK, but the float must be an exact integer:
  # >>> factorial(30.1)
  # Traceback (most recent call last):
  #     ...
  # ValueError: n must be exact integer
  # >>> factorial(30.0)
  # 265252859812191058636308480000000

  # It must also not be ridiculously large:
  # >>> factorial(1e100)
  # Traceback (most recent call last):
  #     ...
  # OverflowError: n too large
  """
  tree_current = Tree()

  # X = data
  # X = np.delete(X, class_column, axis=1)
  X = np.delete(data, class_column, axis=1) # Also possible
  y = data[:, class_column]
  features = [item for item in range(data.shape[1]) if item not in features_ignore and item != class_column]
  # print("features: " , * features)
  
  # Test if all samples have the same label and assign the prediction to that label
  if (np.unique(y).shape[0] == 1):
    tree_current.prediction = y[0]
    tree_current.leaf = True #(I know being True is default, but here it is explicit)
  
  # Test if there are features left, if not, assign the most common (mode)
  elif len(features) == 0:
    #https://stackoverflow.com/questions/10797819/finding-the-mode-of-a-list
    tree_current.prediction = max(set(y), key=y.count)
    tree_current.leaf = True #(I know being True is default, but here it is explicit)
  
  # Since everything else has failed, find the next split
  else:
    tree_current.leaf = False
    
    # Solved using Information Gain. We will study the conditional entropy as 
    # the label entropy is the same for all features at this stage
    # Lower conditional entropy is best
    if (use_IG):
      print("using IG")
      # feature_best = None
      # threshold_best = 0.
      feature_best, threshold_best = best_conditional_entropy(data, class_column, features)

      # if (conditional_entropy_testing < conditional_entropy_current):
      #   conditional_entropy_current = conditional_entropy_testing
      #   feature_best = feature_testing
      #   threshold_best = thres

    # Solved using Gain Ratio
    else:
      node_entropy = entropy(X, y)

    tree_current.feature =  feature_best
    tree_current.threshold =  threshold_best
    features_ignore = np.append(features_ignore, feature_best)
    print("Best feature: ", feature_best, "Creating left tree")
    print("feature_best: ", feature_best, " Threshold_best: ", threshold_best)
    tree_current.left = train(data[data[:,feature_best] < threshold_best], class_column, features_ignore, use_IG)

    print("Best feature: ", feature_best,"features_ignore: ", features_ignore, "Creating right tree")
    tree_current.right = train(data[data[:,feature_best] >= threshold_best], class_column, features_ignore, use_IG)
  return tree_current

def get_entropy(y):
  """Return the entropy of the data.

  Parameters: 
  y (np.array): np.array with the labels of the samples in the set

  Returns: 
  Entropy of the set 
  """
  labels, label_count = np.unique((y), return_counts=1)

  total_samples = sum(label_count)
  entro = 0.

  # Iterate over labels
  for num in label_count:
    # print("num: ", num, " total samples ", total_samples)
    entro = entro + (num/total_samples * np.log2(num/total_samples))
    # print ("entro", entro)

  entro = -entro
  return entro


def conditional_entropy(data, class_column, feature, threshold):
  cond_entropy = 0.0

  data_split = data[data[:, feature] < threshold]
  p = data_split.shape[0]/data.shape[0]
  y = data_split[:,class_column]
  cond_entropy = cond_entropy + get_entropy(y) * p

  data_split = data[data[:, feature] >= threshold]
  p = data_split.shape[0]/data.shape[0]
  y = data_split[:,class_column]
  cond_entropy = cond_entropy + get_entropy(y) * p

  return cond_entropy


def best_conditional_entropy(data, class_column, features):
  """Return the entropy of the data.

  Parameters: 
  y (np.array): np.array with the labels of the samples in the set

  Returns: 
  Entropy of the set 
  """

  print("Inside best_conditional_entropy")

  best_cond_ent = None
  best_feature = None
  best_thres = 0.
  print("data left: {1:d},  Attributes left: {0:d}".format(len(features), data.shape[0]))
  
  for feature in features:
    if feature != class_column:
      data = data[data[:, feature].argsort()] 
      X = data[:, feature] 
      y = data[:, class_column]
      labels_entropy = get_entropy(y)
      # print("System entropy: {1:.2f}, data left: {2:d},  Attributes left: {0:d}".format(len(features), labels_entropy, data.shape[0]))
      labels, label_count = np.unique((y), return_counts=1)
      # total_samples = sum(label_count)
      print("Labels: {0}, label_count: {1}, y[0] {2}".format(labels, label_count, y[0]))

      # Check the possible thresholds
      possible_thresholds = []
      label = y[0]
      for i in range(X.shape[0]):
        if label != y[i]: # Check if the labels are different
          #if they are, store the index
          possible_thresholds = np.append(possible_thresholds, X[i])
          label = y[i]

      # prob_label = label_count/total_samples

      for threshold in possible_thresholds:
        conditional_ent = conditional_entropy(data, class_column, feature, threshold)
        # print("conditional_ent {0}".format(conditional_ent))

        if (best_cond_ent == None or conditional_ent < best_cond_ent):
          # best_cond_ent = conditional_ent
          
          best_cond_ent = conditional_ent
          best_thres = threshold
          best_feature = feature
          # print("best_cond_ent:{0}, cond_ent:{1}, best_feature:{2}, thres:{3}".
                # format(best_cond_ent, conditional_ent, best_feature, best_thres))
  print("feature {0}, best_threshold {1}".format(best_feature, best_thres))
  return best_feature, best_thres







