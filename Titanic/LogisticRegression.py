# Titanic Logistic Regression

# Using cross_val_score to get cross validation score on single model
lr = LogisticRegression(C=20)
lr.fit(X_train,y_train)
scores = cross_val_score(lr, X_train, y_train, cv = 3)
scores
scores.mean()


# GridSearchCV to select hyperparameters:
X_train, X_test, y_train, y_test = train_test_split(
    data_age, survived, test_size=0.2, random_state=1)
    
Cvals = [.00001, .0001, .001, .01, .1, 1, 10, 100, 1000]
#Cvals = [.001, .01, .1] + list(np.arange(1,50,5))
clf = GridSearchCV(LogisticRegression(), {'C': Cvals}, cv = 3);
clf.fit(X_train, y_train);
#sorted(clf.cv_results_.keys())

plt.plot(Cvals, 1-clf.cv_results_['mean_test_score'], Cvals, 1-clf.cv_results_['mean_train_score'])
plt.legend(['CV Error', 'Training Error'])
plt.xlabel('C')
plt.ylabel('Proportion of Incorrect Classfications')
#plt.xscale('log')

print '\n\nThe best value of C is: ' + str(clf.best_params_['C'])
print '\nCV Accuracy: ' + str(clf.best_score_) + ', (' + str(1-clf.best_score_) + ' error)'
print '\nAll training data Accuracy: ' + str(clf.score(X_train,y_train)*100) + '%'
#print(classification_report(y_train, clf.predict(X_train)))
print '\nCV SCore: ' + str(clf.score(X_test,y_test))


# LogisticRegressionCV to select hyperparameter C:
lrcv = LogisticRegressionCV()
lrcv.fit(X_train,y_train);

print '\n\nThe best value of C is: ' + str(lrcv.C_[0])
print '\nAll data score: ' + str(lrcv.score(X_train,y_train))
print '\nHold-out Score: ' + str(lrcv.score(X_test,y_test)) + '%\n'


