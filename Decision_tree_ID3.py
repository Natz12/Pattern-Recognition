

# Import the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import math

import statistics # mode() function a sub-set of the statistics module

verbose = False

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

y = df_wine.iloc[:,0]
df_wine = df_wine.iloc[:,1:]
df_wine[df_wine.shape[1]+1] = y

np_wine = df_wine.values

# Try getting the file online if not found try it locally
try:
  print("Getting tic-tac-toe dataset file from the internet")
  df_ttt = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data',
               header = None) #Load from url
except:
  print("Could not find the tic-tac-toe dataset, getting the file locally")
  df_ttt = pd.read_csv('tic-tac-toe.data',
               header=None) 
else:
  print("tic-tac-toe dataset file loaded from the web")

np_ttt = df_ttt.values
# df_wine[50:100]
# array_wine
# df_ttt[0].unique()
df_wine.rename(columns = {0:'truth'})
df_wine




class Tree:
  def __init__(self, feature = None, is_feature_categorical = None,
               threshold = None, categories = None, is_leaf = None, 
               prediction = None, leaves = None):
    self.feature = feature
    self.is_feature_categorical = False
    self.threshold = None
    self.categories = None
    self.is_leaf = True
    self.prediction = None
    # self.left = None
    # self.right = None
    self.leaves = None



# accepts numpy array
# accepts numpy array
def train_decision_tree(data,  class_column = -1, features_ignore = [], use_IG = True):

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

  # data = df.values
  if class_column == -1:
    class_column = data.shape[1]-1
  # X = np.delete(data, class_column, axis=1) 
  y = data[:, class_column]
  # col = df.columns
  features = [item for item in range(data.shape[1]) if item not in features_ignore and item != class_column]
  # print("features: " , * features)

  
  # Test if all samples have the same label and assign the prediction to that label
  if (np.unique(y).shape[0] == 1):
    tree_current.prediction = y[0]
    tree_current.is_leaf = True #(I know being True is default, but here it is explicit)
  
  # Test if there are features left, if not, assign the most common (mode)
  elif len(features) == 0:
    labels, label_count = np.unique((y), return_counts=1) 
    i = np.argmax(label_count)
    
    #https://stackoverflow.com/questions/10797819/finding-the-mode-of-a-list
    # tree_current.prediction = max(set(y), key=y.count)
    tree_current.prediction = labels[i]
    tree_current.is_leaf = True #(I know being True is default, but here it is explicit)
  
  # Since everything else has failed, find the next split
  else:
    tree_current.is_leaf = False

    labels, label_count = np.unique((y), return_counts=1)
    if label_count == []:
      print("stop")
    try:
      i = np.argmax(label_count)
    except:
      print("")
    tree_current.prediction = labels[i]
  
    feature_best, threshold_best, is_categorical = best_conditional_entropy(data, class_column, features, use_IG)

    if feature_best == None:
      tree_current.is_leaf = True
      return tree_current

    else:

      # verbose
      # print(is_categorical)
      if is_categorical:
        tree_current.threshold = None
        tree_current.categories = np.unique(data[:,feature_best])

      else:
        tree_current.threshold = threshold_best
        tree_current.categories = None


      tree_current.feature = feature_best
    

      tree_current.is_feature_categorical = is_categorical
      features_ignore = np.append(features_ignore, feature_best)

      if feature_best == None:
        print("stop")

      data_splits = data_split(data, feature_best, tree_current.threshold)
      leaves = []
      # verbose = True
      #verbose
      if verbose:
        print(tree_current.threshold)

      for leaf, i in zip(data_splits, range(data_splits.shape[0])):
        
        if verbose:
          print('Leaf number {0:d}'.format(i))
          print('split attribute: ', feature_best)
          print('Feat ignore: ', features_ignore)
        if leaf.shape[0] == 0 :
          print("stop")
        leaves.append(train_decision_tree(leaf, class_column, features_ignore, use_IG))
      tree_current.leaves = list(leaves)

    # print("Best feature: ", feature_best, "Creating left tree")
    # print("feature_best: ", feature_best, " Threshold_best: ", threshold_best)
    # tree_current.left = train_decision_tree(data[data[:,feature_best] < threshold_best], class_column, features_ignore, use_IG)

    # print("Best feature: ", feature_best,"features_ignore: ", features_ignore, "Creating right tree")
    # tree_current.right = train_decision_tree(data[data[:,feature_best] >= threshold_best], class_column, features_ignore, use_IG)
  return tree_current



def data_split(dataSamples, feature, thres):
  if thres is None:
    dataLeaves = []
    vals = np.unique((dataSamples)[:, feature])
    for val in vals:
      dataLeaves.append(dataSamples[dataSamples[:, feature] == val])
  else:
    leftData = dataSamples[dataSamples[:, feature] < thres]
    rightData = dataSamples[dataSamples[:, feature] >= thres]
    dataLeaves = [leftData, rightData]
  dataLeaves = np.asarray(dataLeaves)
  return dataLeaves


def get_entropy(y):
  """Return the entropy of the data.

  Parameters: 
  y (np.array): np.array with the labels of the samples in the set

  Returns: 
  Entropy of the set 
  """
  _, label_count = np.unique((y), return_counts=1)

  total_samples = sum(label_count)
  entro = 0.

  # Iterate over labels
  for num in label_count:
    # print("num: ", num, " total samples ", total_samples)
    entro = entro + (num/total_samples * np.log2(num/total_samples))
    # print ("entro", entro)

  entro = -entro
  return entro

# Get conditional entropy of numerical features based on a threshold for the split of the data
def conditional_entropy_numerical(data, class_column, feature, threshold):
  """Get conditional entropy of numerical features based on a threshold for the split of the data
  """
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


def conditional_entropy_categorical(data, class_column, feature):
  """Get conditional entropy of categorical features
  """
  cond_entropy = 0.0
  feature_data = data[:, feature]
  categories, category_count = np.unique((feature_data), return_counts=1)

  for category in categories:
    data_split = data[data[:, feature] == category]
    p = data_split.shape[0]/data.shape[0]
    y = data_split[:,class_column]
    cond_entropy = cond_entropy + get_entropy(y) * p

  return cond_entropy

# Get split information of numerical features based on a threshold for the split of the data
def split_information_numerical(data, feature, threshold):
  """Get split information of numerical features based on a threshold for the split of the data
  """
  split_info = 0.0

  data_split = data[data[:, feature] < threshold]
  p = data_split.shape[0]/data.shape[0]
  split_info = split_info + (p * np.log2(p))

  data_split = data[data[:, feature] >= threshold]
  p = data_split.shape[0]/data.shape[0]
  split_info = split_info + (p * np.log2(p))

  split_info = -split_info

  return split_info


def split_information_categorical(data, feature):
  split_info = 0.0
  feature_data = data[:, feature]
  categories, category_count = np.unique((feature_data), return_counts=1)

  for category in categories:
    data_split = data[data[:, feature] == category]
    p = data_split.shape[0]/data.shape[0]
    split_info = split_info + (p * np.log2(p))

  split_info = -split_info
  
  return split_info



# Inspired in https://stackoverflow.com/a/16908217/12894766
def is_numerical(item):
  try:
    float(item)
    return True
  except:
    return False

def best_conditional_entropy(data, class_column, features, IG):
  """Return the entropy of the data.

  Parameters: 
  y (np.array): np.array with the labels of the samples in the set

  Returns: 
  Entropy of the set 
  """


  # if verbose:
  # print("Inside best_conditional_entropy")

  best_cond_ent = None
  best_feature = None
  best_thres = 0
  # print("data left: {1:d},  Attributes left: {0:d}".format(len(features), data.shape[0]))
  
  for feature in features:
    data = data[data[:, feature].argsort()] 
    X = data[:, feature] 
    y = data[:, class_column]
    # labels_entropy = get_entropy(y)
    # print("System entropy: {1:.2f}, data left: {2:d},  Attributes left: {0:d}".format(len(features), labels_entropy, data.shape[0]))
    # labels, label_count = np.unique((y), return_counts=1)
    # print("Labels: {0}, label_count: {1}, y[0] {2}".format(labels, label_count, y[0]))
    # total_samples = sum(label_count)
    

    # If the data is numerical:
    if all([is_numerical(item) for item in data[:, feature]]):
      is_categorical = False
   
      # Check the possible thresholds
      possible_thresholds = []
      # print(y)
      label = y[0]
      threshold_problema = 0
      # value = X[0]
      for i in range(X.shape[0]):
        if label != y[i]: # Check if the labels are different
          ## if they are, store the index
          # data_before = X[X >=X[i]]
          # data_after = X[X >=X[i]]

          if possible_thresholds == [] and (X[X >=X[i]].shape[0]==0 or X[X<X[i]].shape[0]==0) :
            # data[data[:, feature] >= threshold]
            label = y[i]
            # print("LOOOK: do nothing stop")
            threshold_problema = X[i]
            # i_problema = i
          else:
            possible_thresholds = np.append(possible_thresholds, X[i])
            label = y[i]
            # value = X[i]
      

      # prob_label = label_count/total_samples

      for threshold in possible_thresholds:
                
        # Solved using Information Gain. We will study the conditional entropy as 
        # the label entropy is the same for all features at this stage
        # Lower conditional entropy is best

        if IG:
          conditional_ent = conditional_entropy_numerical(data, class_column, feature, threshold)
          # print("conditional_ent {0}".format(conditional_ent))
        else:
          # conditional_ent = conditional_entropy_numerical(data, class_column, feature, threshold) / split_information_numerical(data, feature, threshold)
          # X_bool = np.array([x < threshold for x in X])

          intrinsic_value = get_entropy(X<threshold)
          if intrinsic_value == -0.0:
            conditional_ent = np.inf
          else:
            conditional_ent = conditional_entropy_numerical(data, class_column, feature, threshold) / intrinsic_value
          # print("get_entropy boolean: ", get_entropy(X<threshold))

        if best_cond_ent == None:
          best_cond_ent = conditional_ent

        if (conditional_ent < best_cond_ent):
          # print("current_cond_ent: {0}, Better_cond_ent: {1}".format(best_cond_ent,conditional_ent))
          # best_cond_ent = conditional_ent
          
          best_cond_ent = conditional_ent
          best_thres = threshold
          best_feature = feature
          
          is_categorical = False 
          # print("best_cond_ent:{0}, cond_ent:{1}, best_feature:{2}, thres:{3}".
                # format(best_cond_ent, conditional_ent, best_feature, best_thres))
      # print("feature {0}, best_threshold {1}".format(best_feature, best_thres))
      # if best_thres == threshold_problema:
        # print("Stop problema grande")

    # If data is not numerical (categorical)
    else:
      is_categorical = True
      if (IG):
        conditional_ent = conditional_entropy_categorical(data, class_column, feature)
      else:
        intrinsic_value = get_entropy(X)
        if intrinsic_value == -0.0:
            conditional_ent = np.inf
        else:
          conditional_ent = conditional_entropy_categorical(data, class_column, feature) / intrinsic_value

      # print("conditional_ent {0}".format(conditional_ent))
      if best_cond_ent == None:
        best_cond_ent = conditional_ent

      if ((conditional_ent < best_cond_ent)):
        # best_cond_ent = conditional_ent
        
        best_cond_ent = conditional_ent
        best_thres = np.unique(X)
        best_feature = feature
        is_categorical = True

      # print("Not numerical")
      
  return best_feature, best_thres, is_categorical
  
 

  

## Cross Validation

def predict_point(tree, point):
  """Return the prediction of a vector given a tree

  Parameters: 
  tree: tree model to use for the prediciton
  point: vector of features to make the prediction

  Returns: 
  label prediciton of a point
  """

  if (tree.is_leaf):
    return tree.prediction
  i = tree.feature
  # print( "tree.feature: ", i)

  if tree.is_feature_categorical:
    categories = tree.categories
    
    # verbose
    # print("point[i]: {0}\t categories: {1}".format(point[i],categories))
    if point[i] in categories:
      category = np.where(categories == point[i])

      # # verbose
      # print("categor: ", category)
      # print("category[]: ", category[0])
      # print("category index:", int(category[0]))
      return predict_point(tree.leaves[int(category[0])], point)
    else:
      return tree.prediction

    # if i in tree.leaves:
    #   return predict(tree.leaves[i], point)
    # else:
    #   return tree.prediction
  else:
    if (point[i] < tree.threshold):
      return (predict_point(tree.leaves[0], point))
    else:
      return (predict_point(tree.leaves[1], point))

def predict_matrix(tree, data):
  """Return the prediction of a matrix given a tree

  Parameters: 
  tree: tree model to use for the prediciton
  data: samplesto make the prediction

  Returns: 
  vector with prediciton of samples
  """
  prediction = []
  for i in range(data.shape[0]):
    prediction.append(predict_point(tree, data[i]))
  return prediction



def grouper (data, folds = 10):
  """Returns the data with an assigned group number.

  Parameters: 
  data: data to be altered
  folds = number of groups to assign to the data

  Returns: 
  data with an assigned group number
  """
  data_size = data.shape[0]

  group = [*range(folds)]
  group = group*(int(np.floor(data_size/folds)))
  group = group + [*range(data_size%folds)]

  # df = pd.DataFrame(data)
  # df['group']=np.array(group)
  data = np.c_[data, group]
  return data

def ConfusionMatrix(real, predicted):
  classification_matrix = np.c_[real, predicted]

  labels = np.unique(real)
  confusionMatrix = np.zeros((labels.shape[0],labels.shape[0]))
  accuracy = np.full(labels.shape[0],0)
  num_class_sample = np.full(labels.shape[0],0)

  for i in range(labels.shape[0]):
    # j=i+1
    for j in range(labels.shape[0]):
      # l = k+1
      # confusionMatrix[i,j] = real[real == labels[i]]
      confusionMatrix[i,j] = classification_matrix[classification_matrix[:,0]==labels[i]][classification_matrix[classification_matrix[:,0]==labels[i]][:,1]==labels[j]].shape[0]
      num_class_sample[j] = num_class_sample[j] + confusionMatrix[i,j]
      if i == j:
        accuracy[j] = accuracy[j] + confusionMatrix[i,j]

  accuracy = np.round(np.divide(sum(accuracy), sum(num_class_sample)),2)*100  
    
  return confusionMatrix, accuracy, num_class_sample

def get_accuracy(real,predicted):
  classification_matrix = np.c_[real, predicted]
  labels = np.unique(real)
  # confusionMatrix = np.zeros((labels.shape[0],labels.shape[0]))
  accuracy = np.full(labels.shape[0],0)
  # num_class_sample = np.full(labels.shape[0],0)

  for i in range(labels.shape[0]):
    # j=i+1
    for j in range(labels.shape[0]):
      # l = k+1
      # confusionMatrix[i,j] = real[real == labels[i]]
      # confusionMatrix[i,j] = classification_matrix[classification_matrix[:,0]==labels[i]][classification_matrix[classification_matrix[:,0]==labels[i]][:,1]==labels[j]].shape[0]
      # num_class_sample[j] = num_class_sample[j] + confusionMatrix[i,j]
      if i == j:
        accuracy[j] = accuracy[j] + classification_matrix[classification_matrix[:,0]==labels[i]][classification_matrix[classification_matrix[:,0]==labels[i]][:,1]==labels[j]].shape[0]
      

  accuracy = np.round(np.divide(sum(accuracy), real.shape[0]),2)
    
  return accuracy


def k_fold_validation(data, folds = 10, use_IG = True):
  """Cross validation

  Parameters: 
  data: data to be altered
  folds = number of groups to split the data into
  Use_IG = parameter to control the training of the tree. True: Use Information gain. False: Use gain ratio

  Returns: 
  best_accuracy, accuracies, best_testy, best_prediction
  """
  data_group = grouper(data, folds)
  groups = [x for x in range(folds)]

  best_accuracy = 0
  accuracies = []
  best_testy = np.zeros(data.shape[0])
  best_prediction = np.zeros(data.shape[0])

  
  # best_tree = None
  mean_accuracy = 0
  
  prediction = []
  
  for group in groups:
    data_train = data_group[data_group[:,-1] != group]

    data_train = np.delete(data_train, -1, 1)


    data_test = data_group[data_group[:,-1] == group]
    data_test = np.delete(data_test,-1, 1)
    # print("data_test:",data_test.shape)
    testX = data_test[:,:-1]
    testy = data_test[:,-1]

    tree = train_decision_tree(data_train, use_IG = use_IG)

    prediction = predict_matrix(tree, testX)

    # confusion_matrix, accuracy, num_class_sample  = ConfusionMatrix(testy, prediction)

    # print(best_accuracy)
    accuracy = get_accuracy(testy, prediction)
    accuracies.append(accuracy)

    mean_accuracy = mean_accuracy + accuracy/folds

    if accuracy > best_accuracy:
      best_accuracy = accuracy
      # best_tree = tree
      # best_confusion_matrix = confusion_matrix
      best_testy = testy
      best_prediction = prediction
  return best_accuracy, accuracies, best_testy, best_prediction

def multiple_k_fold(data, multiple= 10, folds= 10, use_IG = True):
  """ Multiple Cross validation

  Parameters: 
  data: data to be altered
  multiple: number of k-fold-cross-validation
  folds: number of groups to split the data into
  Use_IG: parameter to control the training of the tree. True: Use Information gain. False: Use gain ratio

  Returns: 
  ###
  """
  # best_confusion_matrix = []
  best_accuracy = 0
  mean_accuracies = []
  for _ in range(multiple):
    data_shuffle = data.copy()
    np.random.shuffle(data)
    accuracy, accuracies, testy, prediction = k_fold_validation(data_shuffle, folds = folds, use_IG = use_IG)
    mean_accuracies +=accuracies
    # mean_accuracies.append(mean_accuracy)
    if accuracy > best_accuracy:
      best_accuracy = accuracy
      # best_confusion_matrix = confusion_matrix
      best_testy = testy
      best_prediction = prediction
  mean_accuracy_variance = np.var(np.array(mean_accuracies))
  total_mean_accuracy = sum(mean_accuracies)/len(mean_accuracies)
  best_confusion_matrix, _, _  = ConfusionMatrix(best_testy, best_prediction)
  return total_mean_accuracy, mean_accuracy_variance, best_accuracy, best_confusion_matrix
  


# tree_wine_IG = train_decision_tree(np_wine, use_IG = True)
# tree_wine = train_decision_tree(np_wine, use_IG = False)


# tree_ttt_IG = train_decision_tree(np_ttt, use_IG = True)
# tree_ttt = train_decision_tree(np_ttt, use_IG = False)
# print((multiple_k_fold(np_ttt)))

multiple_ttt = (multiple_k_fold(np_ttt))
multiple_wine = (multiple_k_fold(np_wine))

multiple_ttt_GR = (multiple_k_fold(np_ttt, use_IG = False))
multiple_wine_GR = (multiple_k_fold(np_wine, use_IG = False))

print(multiple_wine)
print(multiple_ttt)

wine_conf_mat = multiple_wine[3]
ttt_conf_mat = multiple_ttt[3]
wine_conf_mat_GR = multiple_wine_GR[3]
ttt_conf_mat_GR = multiple_ttt_GR[3]

import matplotlib as mplab  #Change document format
sns.set_context("talk")
# plt.style.use('seaborn-talk')

#f5f5ff

mplab.rcParams['figure.figsize'] = [15,9]
fig, ax = plt.subplots(nrows = 1,ncols = 2, sharey = True, figsize = [21.0, 10.0])

# ax[0] = sns.heatmap(confusionMatrixML, annot = True, fmt = 'g', cbar = True, ax = ax[0], square = True, cmap="BuPu", vmin = 5, vmax = 660, center = 150, robust = True)
ax[0] = sns.heatmap(wine_conf_mat, 
                    annot = True, 
                    fmt = 'g', 
                    cbar = True, 
                    ax = ax[0], 
                    square = True, 
                    cmap="BuPu", 
                    vmin= -2, 
                    vmax = 8,
                    robust = True,
                    annot_kws={ "weight": "bold"})
ax[0].set_xlabel('(a) Information Gain')
ax[0].set_title("Real Values")
ax[0].set_ylabel("Predicted Values")

# ax[1] = sns.heatmap(confusionMatrixMAP, annot = True, fmt = 'g', cbar = True, ax = ax[1], square = True, cmap="PuBu", vmin = 5, vmax = 660, center = 150, robust = True) #"RdYlGn" cubehelix
ax[1] = sns.heatmap(wine_conf_mat_GR, 
                    annot = True, 
                    fmt = 'g', 
                    cbar = True, 
                    ax = ax[1], 
                    square = True, 
                    cmap="BuPu", 
                    vmin = -2,
                    vmax = 8,
                    # center = 900, 
                    robust = True,
                    annot_kws={ "weight": "bold"}) #"RdYlGn" cubehelix
ax[1].set_xlabel('(b) Gain Ratio')
ax[1].set_title("Real Values")
fig.tight_layout()

fig, ax = plt.subplots(nrows = 1,ncols = 2, sharey = True, figsize = [21.0, 10.0])

# ax[0] = sns.heatmap(confusionMatrixML, annot = True, fmt = 'g', cbar = True, ax = ax[0], square = True, cmap="BuPu", vmin = 5, vmax = 660, center = 150, robust = True)
ax[0] = sns.heatmap(ttt_conf_mat, 
                    annot = True, 
                    fmt = 'g', 
                    cbar = True, 
                    ax = ax[0], 
                    square = True, 
                    cmap="PuBu", 
                    vmin = -20,
                    vmax = 69, 
                    robust = True,
                    annot_kws={ "weight": "bold"})
ax[0].set_xlabel('(a) Information Gain')
ax[0].set_title("Real Values")
ax[0].set_ylabel("Predicted Values")

# ax[1] = sns.heatmap(confusionMatrixMAP, annot = True, fmt = 'g', cbar = True, ax = ax[1], square = True, cmap="PuBu", vmin = 5, vmax = 660, center = 150, robust = True) #"RdYlGn" cubehelix
ax[1] = sns.heatmap(ttt_conf_mat_GR, 
                    annot = True, 
                    fmt = 'g', 
                    cbar = True, 
                    ax = ax[1], 
                    square = True, 
                    cmap="PuBu", 
                    vmin = -20,
                    vmax = 69,
                    # center = 900, 
                    robust = True,
                    annot_kws={ "weight": "bold"}) #"RdYlGn" cubehelix
ax[1].set_xlabel('(b) Gain Ratio')
ax[1].set_title("Real Values")
fig.tight_layout()






  

  