from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the Iris dataset
iris = load_iris()

# The features (X) are the sepal and petal lengths and widths
X = iris.data

# The labels (y) are the species names
y = iris.target

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier on the training set
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(clf, 'iris_classifier.joblib')

# Predict the labels for the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
