import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("../../data_sources/gestures.csv")

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
plt.title("Macierz pomyłek DTC")
plt.savefig("dtc_extended_confusion_matrix.png")
plt.show()


with open('../../models/DTC/DTC_extended.pkl', 'wb') as f:
    import pickle
    pickle.dump(clf, f)
print("Model zapisany do 'DTC.pkl'")





