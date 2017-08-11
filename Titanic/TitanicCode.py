import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
from subprocess import check_output
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from scipy import interp
import re

dir = '/Users/stevenfelix/Dropbox/DataScience/Projects/Kaggle - Titanic/'
data = pd.read_csv(dir+"train.csv")
data.head()


####### Feature Engineering  #########


# Extracting and Cleaning up titles info:
re_title = re.compile(pattern = '(?<= )\w+\.')
data['Title'] = [re_title.search(name).group(0) for name in data.Name if re_title.search(name)]
#title_count = data.Title.value_counts()
#title_count

data.loc[data.Title == 'Mme.', 'Title'] = 'Mrs.'
data.loc[data.Title == 'Ms.', 'Title'] = 'Miss.'
data.loc[data.Title == 'Mlle.', 'Title'] = 'Miss.'
#data.loc[data.Title == 'Lady.', 'Title'] = 'Miss.'
#data.loc[data.Title == 'Countess.', 'Title'] = 'Mrs.'
military = ['Col.','Major.','Capt.']
Hon = ['Don.','Jonkheer.','Sir.', 'Lady.','Countess.']
data.loc[[i in military for i in data.Title], 'Title'] = 'Mil.'
data.loc[[i in Hon for i in data.Title], 'Title'] = 'Hon.'
#title_count = data.Title.value_counts()
#title_count

# Extracting last name
re_lastname = re.compile(pattern = "[ a-zA-Z\\']+")
data['LastName'] = [re_lastname.search(name).group(0) for name in data.Name if re_lastname.search(name)]
#data['LastName'].value_counts()

# Cabin
re_cabin = re.compile(pattern = "[ABCDEFG]")
data.Cabin.fillna('Unknown', inplace=True)
data['Deck'] = [re_cabin.search(cabin[0]).group(0) if (cabin and re_cabin.search(cabin[0])) else cabin for cabin in data.Cabin]
#data['Deck'].value_counts()

# Maiden Name
re_maiden = re.compile(pattern = "\w+(?=\\))")
data['Maiden'] = [re_maiden.search(name).group(0) if re_maiden.search(name) else 'NA' for name in data.Name]



########## Data Pre-Processing for Modeling ###########

data_orig = data.copy()

data.columns

# Dummy Variables: Title, Sex
titles = list(set(data.Title))
tmp = preprocessing.label_binarize(data.Title, classes=titles)
tmp = pd.DataFrame(tmp[:,1:], columns=titles[1:]) # must remove 1 column for proper dummy coding
data = data.join(tmp)
data['Sex_female'] = preprocessing.label_binarize(data.loc[:,'Sex'], 
                                                          classes=['male','female']) # female = 1
ytrain = data_dummies['Survived']

data_full.columns

# Interaction Variables
data_dummies['Classex'] = data_dummies['Sex_female']*data_dummies['Pclass']
data_dummies['SibSpsex'] = data_dummies['Sex_female']*data_dummies['SibSp']
data_dummies['AgeSex'] = data_dummies['Sex_female']*data_dummies['Age']
data_dummies['AgeClass'] = data_dummies['Pclass']*data_dummies['Age']
#X_scaled = preprocessing.scale(data_dummies[varnames])


# Imputing Age
impute_missing = data_dummies
impute_missing_cols = list(impute_missing)
print(impute_missing_cols)
filled_MICE = fancyimpute.MICE().complete(np.array(impute_missing));
results = pd.DataFrame(filled_MICE, columns = impute_missing_cols)
#results['Train'] = list(data['Train'])
#results['Survived'] = list(data['Survived'])




############### Visualizations ######################

#  Viz: Survival ~ Title:
labs = [title+' ('+str(count)+')'for title,count in zip(title_count.index, title_count)]
ax = sns.barplot(x = "Title", y = "Survived",data = data, order=title_count.index);
ax.set_xticklabels(labs, rotation = 90);
ax.get_figure().savefig(dir+'survivalTitles2.png')


## Viz: Survival ~ Class/Sex,  Siblings/Sex, ParentCh/Sex, Embarked/Sex, Embarked/Class:
sns.set(font_scale= 2)
fig, axs = plt.subplots(3,2, figsize = (15,20))
sns.barplot(x = "Pclass", y = "Survived", hue = "Sex", data = data, ax = axs[0,0])
sns.barplot(x = "SibSp", y = "Survived", hue = "Sex",data = data, ax = axs[0,1])
sns.barplot(x = "Parch", y = "Survived", hue = "Sex",data = data, ax = axs[1,0])
sns.barplot(x = "Embarked", y = "Survived", hue = "Sex",data = data, ax = axs[1,1])
sns.countplot(x = "Embarked", hue = "Pclass", data = data, ax = axs[2,0])
sns.barplot(x = "Embarked", y = "Fare", hue = "Pclass",data = data, ax = axs[2,1])
fig.savefig(dir+'survivalFacetPlot.png')

# Binning Age For Viz:
b = np.arange(0, 81, 5) # bins
agelabels = [str(v+1)+'-'+str(b[i+1]) for i,v in enumerate(b[:-1])] # labels
data['Age2'] = pd.cut(data['Age'], bins=b, labels= agelabels)
data.Age2 = data.Age2.cat.reorder_categories(agelabels, ordered=True)



# Viz: Survival ~ Age (* sex, * Class):
fig, ax = plt.subplots(figsize = (10,5))
sns.barplot(x = 'Age2', y = 'Survived', data=data, ax = ax, color='lightblue')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90);

fig, ax = plt.subplots(figsize = (15,5))
sns.barplot(x = 'Age2', y = 'Survived', hue = 'Sex', data=data, ax = ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90);


g = sns.factorplot(x = 'Age2', y = 'Survived', data = data, row = 'Pclass', kind = 'bar', aspect=2.5)
g.set_xticklabels(rotation = 90)

# Mssing Data:
data.isnull().sum() # only missing data for Age, Cabin, and embarked
data['AgeMissing'] = data.Age.isnull()


# Viz: Missingness ~ Class:
sns.set(font_scale= 2)
fig, axs = plt.subplots(2,1, sharey=True, figsize = (15,10))
sns.barplot(x = 'Pclass',y = 'AgeMissing', hue = 'Sex', data =data, ax = axs[0])
axs[0].set(xlabel='Class', ylabel='Proportion of data missing')
sns.barplot(x = 'Title',y = 'AgeMissing', data =data, ax = axs[1])
#sns.plt.title('Proportions of missing age \ndata by class') # can also get the figure from plt.gcf()
fig.suptitle('Proportions of missing age \ndata by class')
#People in 3rd class much more likely to have missing data. No difference by Sex


# Viz: Age Distributions by Class:
g = sns.factorplot(x = 'Age2', col = 'Pclass', data = data, kind="count")
g.set(xticks=[])
print("Average ages: \n", data.groupby('Pclass').Age.mean())
## Third class appears to be younger
