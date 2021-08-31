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
from bokeh.models import Legend, HoverTool
from bokeh.palettes import cividis

from bokeh.io import curdoc
from bokeh.themes import built_in_themes

# Classifiers
from sklearn.tree import DecisionTreeClassifier

# Model Selection / Pre-processing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import learning_curve, validation_curve

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV

from scipy.stats import randint
from scipy.stats.kde import gaussian_kde

# metrics
from sklearn import metrics

curdoc().theme = 'dark_minimal'

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
def hyper_grid_search(estimator, data, scoring, param_grid, cv):
  gridsearch = HalvingGridSearchCV(estimator=estimator, param_grid=param_grid, 
                                 cv=cv, scoring=scoring)
  gridsearch.fit(data.X_train, data.y_train)

  return gridsearch.best_params_, pd.DataFrame(gridsearch.cv_results_)

''' Hyperparameter Random Search '''
def hyper_randgrid_search(estimator, data, scoring, param_dist, cv):
  gridsearch = HalvingRandomSearchCV(estimator=estimator, 
                                     param_distributions=param_dist,
                                     cv=cv, scoring=scoring)
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
def distplot(data, title_str, bins=10):
  hist, edges = np.histogram(data, density=True, bins=bins)
  x = np.linspace(min(data), max(data), 200)
  pdf = gaussian_kde(data)
  
  p = figure(plot_width=300, plot_height=300,
             tools='reset, box_zoom', 
             background_fill_color="#fafafa")
  p.toolbar.autohide = True
  p.title.text = title_str
  p.title.text_font_size = "10px"
  
  TOOLTIPS_bar = [('Error', '@left'),
                 ('Prob', '@prob')]
  TOOLTIPS_pdf = [('Error', '@x'),
                 ('Prob_PDF', '@pdf')]
  
  cdsbar = ColumnDataSource(data={'prob':hist,
                                  'left':edges[:-1],
                                  'right':edges[1:]})
  cdspdf = ColumnDataSource(data={'x':x,
                                  'pdf':pdf(x)})

  bar_p = p.quad(top='prob', bottom=0, left='left', right='right',
          fill_color="navy", line_color="white", alpha=0.5, source=cdsbar)
  p.add_tools(HoverTool(renderers=[bar_p], tooltips=TOOLTIPS_bar, mode='vline'))
  
  pdf_p = p.line(x='x', y='pdf', line_color="#ff8888", 
                 line_width=4, alpha=0.7, source=cdspdf)
  p.add_tools(HoverTool(renderers=[pdf_p], tooltips=TOOLTIPS_pdf, mode='vline'))

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
  
  title = f"{type(estimator).__name__}'s {scoring} variance"
  distplot(scores, title, bins=bins)
  return

''' Plot Feature Importance '''
def plot_feature_importance(estimator, data):
  importances = estimator.feature_importances_

  indices = np.argsort(importances)[::-1]

  # Names points to data.keys()?
  names   = data.raw_data.keys().tolist()
  f_names = [names[i] for i in indices]

  p = figure(plot_width=600, plot_height=300, tools='reset, box_zoom')
  p.toolbar.autohide = True
  
  p.title.text = f'Feature importance: {type(estimator).__name__}'
  p.title.text_font_size = '10px'
  
  TOOLTIPS_bar = [('Importance', '@Importance'),
                 ('Feature', '@FeatureLabel')]
  
  label_dict = {}
  for i, s in enumerate(f_names):
    label_dict[i] = s
  
  cds = ColumnDataSource(data={'Importance':importances[indices],
                               'Feature':range(data.X_train.shape[1]),
                               'FeatureLabel':f_names})

  bar_p = p.vbar(x='Feature', width=0.5, bottom=0, top='Importance', source=cds)
  p.add_tools(HoverTool(renderers=[bar_p], tooltips=TOOLTIPS_bar, mode='vline'))

  p.xaxis.major_label_overrides = label_dict
  p.xaxis.major_label_orientation = np.pi/4
  p.xaxis.ticker = list(label_dict.keys())

  show(p)
  return

''' Plotting Learning Curves '''
def plot_learning_curves(estimator, train_size, data, cv):
  (train_sizes, train_scores, 
   test_scores) = learning_curve(estimator,
                                 data.X_train, data.y_train,
                                 cv=cv, n_jobs=-1,
                                 train_sizes=np.linspace(.1, 1.0, train_size))
        
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

  p = figure(plot_width=600, plot_height=300, tools='reset, box_zoom')
  p.toolbar.autohide = True
  
  p.title.text = f'Learning curves: {type(estimator).__name__}'
  p.title.text_font_size = '10px'
  
  colors = cividis(10)

  p.varea(x='train_sizes', y1='train_scores_lower', y2='train_scores_upper',
          fill_color=colors[1], fill_alpha=0.5, source=cds)
  
  p.varea(x='train_sizes', y1='test_scores_lower', y2='test_scores_upper',
          fill_color=colors[3], fill_alpha=0.5, source=cds)
  
  TOOLTIPS_tr = [('Train Score', '@train_scores_mean'),
                 ('#samples', '@train_sizes')]
  TOOLTIPS_te = [('Test Score', '@test_scores_mean'),
                 ('#samples', '@train_sizes')]
  
  tr_p = p.line(x='train_sizes', y='train_scores_mean', source=cds,
               line_color=colors[-1])
  p.add_tools(HoverTool(renderers=[tr_p], tooltips=TOOLTIPS_tr, mode='vline'))
  
  te_p = p.line(x='train_sizes', y='test_scores_mean', source=cds,
               line_color=colors[-2])
  p.add_tools(HoverTool(renderers=[te_p], tooltips=TOOLTIPS_te, mode='vline'))
  
  legend = Legend(items=[('Training score'   , [tr_p]),
                         ('Cross-validation score' , [te_p])], 
                  location="center")

  p.add_layout(legend, 'right')
  p.legend.label_text_font_size = '9pt'
  p.legend.click_policy="hide"
  
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

  p = figure(plot_width=600, plot_height=300, tools='reset, box_zoom')
  p.toolbar.autohide = True
  
  p.title.text = f'Validation curves for {param_dict['param_name']} on {type(estimator).__name__} with {scoring}'
  p.title.text_font_size = '10px'
  
  colors = cividis(10)

  p.varea(x='param_range', y1='train_scores_lower', y2='train_scores_upper',
          fill_color=colors[2], fill_alpha=0.7, source=cds)
  
  p.varea(x='param_range', y1='test_scores_lower', y2='test_scores_upper',
          fill_color=colors[4], fill_alpha=0.7, source=cds)
  
  TOOLTIPS_tr = [('Train Score', '@train_scores_mean'),
                 (f"{param_dict['param_name']}", '@param_range')]
  TOOLTIPS_te = [('Test Score', '@test_scores_mean'),
                 (f"{param_dict['param_name']}", '@param_range')]
  
  tr_p = p.line(x='param_range', y='train_scores_mean', source=cds, 
         line_color=colors[-3])
  p.add_tools(HoverTool(renderers=[tr_p], tooltips=TOOLTIPS_tr, mode='vline'))
  
  te_p = p.line(x='param_range', y='test_scores_mean', source=cds, 
         line_color=colors[-4])
  p.add_tools(HoverTool(renderers=[te_p], tooltips=TOOLTIPS_te, mode='vline'))
  
  legend = Legend(items=[('Training score'   , [tr_p]),
                         ('Cross-validation score' , [te_p])], 
                  location="center", label_text_font_size = '9pt')

  p.add_layout(legend, 'right')
  p.legend.click_policy="hide"
  
  show(p)
  return

print("Finished loading andresML 🤓👌🏽")