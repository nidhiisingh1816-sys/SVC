from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

# Load dataset (Iris)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create SVC model
model = SVC(
    kernel='rbf',   # 'linear', 'poly', 'sigmoid' also possible
    C=1.0,
    gamma='scale'
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))