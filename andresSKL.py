#<--------- ML Module for SKLearn ------>
#<--------- Andres Cardenas, 2021 ------>

''' IMPORTS '''
# Basics
import numpy as np
import pandas as pd
import pandas_bokeh
import matplotlib.pyplot as plt

# Bokeh Plotting
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models import TickFormatter

# Classifiers
from sklearn.tree import DecisionTreeClassifier

# Model Selection / Pre-processing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import learning_curve

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV

from scipy.stats import randint
from scipy.stats.kde import gaussian_kde

# metrics
from sklearn import metrics

''' OOP '''
class ModelData:
  def __init__(self, raw_data, test_size=0.2):
    self.test_size = test_size
    self.raw_data  = raw_data

    datas = data_split(self.raw_data, self.test_size)
    self.X_train = datas[0]
    self.X_test  = datas[1]
    self.y_train = datas[2]
    self.y_test  = datas[3]

    def change_split(self,test_size):
      self.test_size = test_size
      datas = data_split(self.raw_data, self.test_size)
      self.X_train = datas[0]
      self.X_test  = datas[1]
      self.y_train = datas[2]
      self.y_test  = datas[3]
        
''' Data Pre-processing '''
def data_split(raw_data, test_size):
  train_data, test_data = train_test_split(raw_data, 
                                           test_size=test_size,
                                           stratify=raw_data['label'])
  X_train = train_data.drop('label', axis = 1)
  X_test  = test_data.drop('label', axis = 1)
  y_train = train_data.label
  y_test  = test_data.label
  return X_train, X_test, y_train, y_test

''' -------- METRICS --------- '''

''' Cross-Validation '''
def crossval(estimator, data, cv):
  scoring = ['precision', 'recall', 'f1', 'accuracy']
  scores = cross_validate(estimator,
                          data.X_train, data.y_train,
                          scoring=scoring, cv=cv)
  print("Scores:")
  for metric in scoring:
    met_key = 'test_' + metric
    print(f"{metric}: {scores[met_key].mean():.2f} accuracy with a standard deviation of {scores[met_key].std():.2f}\n")
  return

''' Hyperparameter Grid Search '''
def hyper_grid_search(estimator, data, param_grid, cv):
  gridsearch = HalvingGridSearchCV(estimator=estimator, param_grid=param_grid, 
                                 cv=cv, scoring='accuracy')
  gridsearch.fit(data.X_train, data.y_train)

  return gridsearch.best_params_, pd.DataFrame(gridsearch.cv_results_)

''' Hyperparameter Random Search '''
def hyper_randgrid_search(estimator, data, param_dist, cv):
  gridsearch = HalvingRandomSearchCV(estimator=estimator, 
                                     param_distributions=param_dist,
                                     cv=cv, scoring='accuracy')
  gridsearch.fit(data.X_train, data.y_train)

  return gridsearch.best_params_, pd.DataFrame(gridsearch.cv_results_)

''' Final Score (Classifiers) '''
def model_score(params, data, clf):
  clf.set_params(**params)
  clf.fit(data.X_train, data.y_train)
  y_pred = clf.predict(data.X_test)
  target_names = [] # Modify for specific model
  print(metrics.classification_report(data.y_test,
                                      y_pred, target_names=target_names))
  return y_pred


''' -------- VISUALIZATION --------- '''
''' Plot Distributions '''
def distplot(data, title, bins=10):
  hist, edges = np.histogram(data, density=True, bins=bins)
  x = np.linspace(min(data), max(data), 200)
  pdf = gaussian_kde(data)
  
  p = figure(plot_width=300, plot_height=300,
             tools='reset, box_zoom, hover', 
             background_fill_color="#fafafa")
  p.toolbar.autohide = True
  p.hover.mode = 'vline'
  p.title = title  

  p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
          fill_color="navy", line_color="white", alpha=0.5)
  p.line(x, pdf(x), line_color="#ff8888", line_width=4, alpha=0.7)

  p.y_range.start = 0
  p.xaxis.axis_label = 'x'
  p.yaxis.axis_label = 'Pr(x)'
  p.grid.visible = False

  show(p)
  return

''' Model Variance '''
def plot_model_variance(estimator, data, scoring, bins, train=True):
  sss_split = StratifiedShuffleSplit(n_splits = 100 , test_size=0.75)
  if train==True:
    X = data.X_train
    y = data.y_train
  else:
    X = data.X_test
    y = data.y_test
  scores = cross_val_score(estimator, X, y, scoring=scoring, cv=sss_split)
  
  title = f"{type(estimator).__name__}'s {scoring} score variance"
  print(title)
  distplot(scores, title, bins=bins)
  return

''' Plot Feature Importance '''
def plot_feature_importance(estimator, data):
  importances = estimator.feature_importances_

  indices = np.argsort(importances)[::-1]

  # Names points to data.keys()?
  # names = data.keys()
  f_names = [names[i] for i in indices]

  p = figure(plot_width=300, plot_height=300, tools='reset, box_zoom, hover')
  p.toolbar.autohide = True
  p.hover.mode = 'vline'

  p.vbar(x=range(data.X_train.shape[1]),
         width=0.5, bottom=0, top=importances[indices])
  
  label_dict = {}
  for i, s in enumerate(f_names):
    label_dict[i] = s

  p.xaxis.formatter = TickFormatter(labels=label_dict)
  p.xaxis.major_label_orientation = pi/4

  show(p)
  return

''' Plotting Learning Curves '''
def plot_learning_curves(estimator, train_num, data, cv):
  (train_sizes, train_scores, 
   test_scores) = learning_curve(estimator,
                                 data.X_train, data.y_train,
                                 cv=cv, n_jobs=-1,
                                 train_sizes=np.linspace(.1, 1.0, train_num))
        
  data = {'train_scores_mean': np.mean(train_scores, axis=1),
          'train_scores_std' : np.std(train_scores, axis=1),
          'test_scores_mean' : np.mean(test_scores, axis=1),
          'test_scores_std'  : np.std(test_scores, axis=1),
          'train_sizes'      : train_sizes}
  
  data['train_scores_upper'] = data['train_scores_mean'] + data['train_scores_std']
  data['train_scores_lower'] = data['train_scores_mean'] - data['train_scores_std']
  data['test_scores_upper']  = data['test_scores_mean']  + data['test_scores_std']
  data['test_scores_lower']  = data['test_scores_mean']  - data['test_scores_std']

  cds = ColumnDataSource(data)

  p = figure(plot_width=400, plot_height=400, tools='reset, box_zoom, hover')
  p.toolbar.autohide = True
  p.hover.mode = 'vline'

  p.varea(x='train_sizes', y1='train_scores_lower', y2='train_scores_upper',
          fill_alpha=0.7, source=cds)
  
  p.varea(x='train_sizes', y1='test_scores_lower', y2='test_scores_upper',
          fill_alpha=0.7, source=cds)
  
  p.line(x='train_sizes', y='train_scores_mean', source=cds, 
         legend_name='Training score')
  
  p.line(x='train_sizes', y='test_scores_mean', source=cds, 
         legend_name='Cross-validation score')
  
  show(p)
  return

''' Plotting Validation Curves '''
def plot_val_curves(estimator, param_dict, scoring, data, cv):
  train_scores, test_scores =  validation_curve(estimator, data.X_train, data.y_train,
                                                param_name=param_dict['param_name'],
                                                param_range=param_dict['param_range'],
                                                scoring=scoring, n_jobs=-1)
  
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std  = np.std(train_scores, axis=1)
  test_scores_mean  = np.mean(test_scores, axis=1)
  test_scores_std   = np.std(test_scores, axis=1)

  data = {'train_scores_mean': np.mean(train_scores, axis=1),
          'train_scores_std' : np.std(train_scores, axis=1),
          'test_scores_mean' : np.mean(test_scores, axis=1),
          'test_scores_std'  : np.std(test_scores, axis=1),
          'param_range'      : param_dict['param_range']}
  
  data['train_scores_upper'] = data['train_scores_mean'] + data['train_scores_std']
  data['train_scores_lower'] = data['train_scores_mean'] - data['train_scores_std']
  data['test_scores_upper']  = data['test_scores_mean']  + data['test_scores_std']
  data['test_scores_lower']  = data['test_scores_mean']  - data['test_scores_std']

  cds = ColumnDataSource(data)

  p = figure(plot_width=400, plot_height=400, tools='reset, box_zoom, hover')
  p.toolbar.autohide = True
  p.hover.mode = 'vline'

  p.varea(x='param_range', y1='train_scores_lower', y2='train_scores_upper',
          fill_alpha=0.7, source=cds)
  
  p.varea(x='param_range', y1='test_scores_lower', y2='test_scores_upper',
          fill_alpha=0.7, source=cds)
  
  p.line(x='param_range', y='train_scores_mean', source=cds, 
         legend_name='Training score')
  
  p.line(x='param_range', y='test_scores_mean', source=cds, 
         legend_name='Cross-validation score')
  show(p)
  return
print("Finished loading andresML 🤓👌🏽")