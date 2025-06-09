import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("../data_sources/gestures.csv")

X = df.drop("label", axis=1).values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

print("Accuracy:", clf.score(X_test, y_test))
print("Cross-validation scores:", cross_val_score(clf, X, y, cv=5))
print("Classification Report:")
print(classification_report(y_test, clf.predict(X_test)))

cm = confusion_matrix(y_test, clf.predict(X_test), labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title("Macierz pomyłek")
plt.savefig("dtc_confusion_matrix.png")
plt.show()


with open('../models/DTC/DTC.pkl', 'wb') as f:
    import pickle
    pickle.dump(clf, f)
print("Model zapisany do 'DTC.pkl'")



# Accuracy: 0.9913606911447084
# Cross-validation scores: [0.98058252 0.98701299 0.99675325 0.98376623 0.96103896]
# Classification Report:
#               precision    recall  f1-score   support

#            L       0.99      1.00      1.00       140
#            V       1.00      0.99      1.00       131
#           ok       0.94      1.00      0.97        51
#         palm       1.00      0.98      0.99       141

#     accuracy                           0.99       463
#    macro avg       0.98      0.99      0.99       463
# weighted avg       0.99      0.99      0.99       463



