# Fitting Logistic Regression to the Training set (requires feature scaling)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#Producing Confusion Matrix
y_pred = classifier.predict(X_test_1)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_1, y_pred)

#Evaluating Confusion Matrix
from sklearn.metrics import precision_recall_fscore_support
y_true = y_test_1
Eval = precision_recall_fscore_support(y_true, y_pred, average = None)