## Titanic: Random Forests

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
from collections import OrderedDict
import fancyimpute

dir = '/Users/stevenfelix/Dropbox/DataScience/Projects/Kaggle - Titanic/'
data = pd.read_csv(dir+"train.csv").assign(Training = 1)
testdata = pd.read_csv(dir+"test.csv").assign(Survived = -99).assign(Training = 0)[list(data)]
data = pd.concat([data,testdata], ignore_index=True)


########## Data Pre-Processing for Modeling ###########

### Create new vars:

# Title
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
Hon = ['Don.','Jonkheer.','Sir.', 'Lady.','Countess.','Dona.']
data.loc[[i in military for i in data.Title], 'Title'] = 'Mil.'
data.loc[[i in Hon for i in data.Title], 'Title'] = 'Hon.'
title_count = data.Title.value_counts()
#title_count

# Last Name
re_lastname = re.compile(pattern = "[ a-zA-Z\\']+")
data['LastName'] = [re_lastname.search(name).group(0) for name in data.Name if re_lastname.search(name)]
#data['LastName'].value_counts()

# Deck
re_cabin = re.compile(pattern = "[ABCDEFG]")
data.Cabin.fillna('Unknown', inplace=True)
data['Deck'] = [re_cabin.search(cabin[0]).group(0) if (cabin and re_cabin.search(cabin[0])) else cabin for cabin in data.Cabin]
data.loc[data.Deck=='T','Deck'] = 'Unknown'
#data['Deck'].value_counts()

# Maiden Name
re_maiden = re.compile(pattern = "\w+(?=\\))")
data['Maiden'] = [re_maiden.search(name).group(0) if re_maiden.search(name) else 'NA' for name in data.Name]
data['Maiden'].value_counts()

# Family size
data['FamilySize'] = data.Parch + data.SibSp

####

# Dummy coding
data[['Title', 'LastName', 'Maiden','Pclass','Sex','Deck']].isnull().sum() # no NAs
data = pd.get_dummies(data, columns=['Title', 'Pclass','Sex','Deck','Embarked'])
#data = pd.get_dummies(data, columns=['LastName'])
#pd.get_dummies(dataRF['Sex'], dummy_na=True) # returns dummies for all variables given
#data['Sex'].pipe(pd.get_dummies, dummy_na=True) # works same as above

####

# Droping unneeded variables
outcomes = data.Survived
Training = data['Training']
data.drop(['PassengerId','Survived','Name','SibSp','Parch','Ticket','Fare',
            'Training','Cabin','Maiden','LastName'], axis = 1, inplace=True)


# Imputing Age
def impute(df):
    return pd.DataFrame(fancyimpute.MICE().complete(np.array(df)), columns = list(df))

data_full = impute(data).join(Training)

# Cacluating Interactions from dummy variables

def dummyinteractions(df, varstems):
    df = df.copy()
    namesdict = getnames(varstems, list(df))
    for i, stem1 in enumerate(varstems[:-1]):
        for stem2 in varstems[i+1:]:
            df = multiply(df, namesdict[stem1], namesdict[stem2])
    return df

def multiply(df, list1, list2):
   for var1 in list1:
       for var2 in list2:
           newname = var1+'_'+var2
           df[newname] = df[var1]*df[var2]
   return df

def getnames(stems, varlist):
    d = {}
    for stem in stems:
        d[stem] = [var for var in varlist if var.startswith(stem)]
    return d

interactionvars = ['Pclass','Sex','FamilySize', 'Age']
data_full = dummyinteractions(data_full, interactionvars)

#### splitting off test data
submissiondat = data_full[data_full.Training == 0].drop(['Training'], axis = 1)
data_full = data_full[data_full.Training == 1].drop(['Training'], axis = 1)


#### 

# Scaling Data
#X_scaled = preprocessing.scale(data_full)

###### Split Data #############
outcomes = outcomes[outcomes != -99]

X_train, X_test, y_train, y_test = train_test_split(
            data_full, outcomes, test_size = 0.1, random_state = 50)


####### Random Forest ############


rfc = RandomForestClassifier(n_estimators=200, max_features = 'log2', min_samples_leaf = 3,
                            oob_score= True, random_state=1)
rfc.fit(X_train,y_train)

rfc.oob_score_  # meh .83
rfc.score(X = X_train,y = y_train) # PERFECT --> overfit
rfc.score(X = X_test,y = y_test) # 'Hold out' / test accuracy, meh, .79
np.mean(cross_val_score(rfc, X_scaled, outcomes, cv = 5)) # overall 5-fold CV accuracy, .83

######## GridSearchCV ###########

rfcGS = RandomForestClassifier(random_state=1)
gridparams = {'n_estimators':[100,200,300],
                'min_samples_split': [2, 3, 6],
                'min_samples_leaf': [1, 3, 5],
                'max_features': ['auto','log2']} 

clf = GridSearchCV(rfcGS, gridparams, cv = 5)
clf.fit(X_train, y_train)

print '\nOptimized parameters: '
for k in clf.best_params_:
    print k + ': ' + str(clf.best_params_[k])
print '\nModel accuracy (hold-out): %.3f' %  clf.score(X_test, y_test)

clf.score(X_train, y_train) # almost perfect 
clf.score(X_test, y_test) # Hold out score comparatively low -- overfitting
clf.score(data_full, outcomes) # goes down from perfect, overfitting
np.mean(cross_val_score(clf.best_estimator_,data_full, outcomes, cv = 5))

##### Re-optimize using full training dataset


clf.fit(data_full,outcomes)
clf.best_params_
clf.best_score_

submissiondf = pd.read_csv(dir+'test.csv')[['PassengerId']].assign(Survived = clf.predict(submissiondat))
submissiondf.head()

submissiondf.to_csv(dir+'submission.csv',index=False)

##################

RANDOM_STATE = 123

# Generate a binary classification dataset.
X, y = X_train, y_train

# NOTE: Setting the `warm_start` construction parameter to `True` disables
# support for parallelized ensembles but is necessary for tracking the OOB
# error trajectory during training.
ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(warm_start=True, oob_score=True,
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features=None",
        RandomForestClassifier(warm_start=True, max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))
]

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of `n_estimators` values to explore.
min_estimators = 175
max_estimators = 250

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X, y)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()

rfc = RandomForestClassifier(n_estimators=22, max_features='log2', oob_score= True, random_state=RANDOM_STATE)
rfc.fit(X_train,y_train)
rfc.oob_score_
rfc.score(X = X_train,y = y_train) # overfitting!
rfc.score(X = X_test,y = y_test)

## Evaluation:  These models have high variance and low bias --- it's overfitting like crazy
