import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import graphviz
import os

df = pd.read_csv("../../data_sources/gestures.csv")

X = df.drop("label", axis=1).values
y = df["label"].values


feature_names = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]


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
plt.savefig("delete.png")
plt.show()


try:
    dot_data = export_graphviz(clf,
                              out_file=None,
                              feature_names=feature_names,
                              class_names=clf.classes_,
                              filled=True,
                              rounded=True,
                              special_characters=True,
                              max_depth=4)
    
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree_graphviz", format='png', cleanup=True)
    print("Graphviz tree saved as 'decision_tree_graphviz.png'")
except:
    print("Graphviz not available - skipping advanced visualization")




with open('../../models/DTC/delete.pkl', 'wb') as f:
    import pickle
    pickle.dump(clf, f)
print("Model zapisany do 'DTC.pkl'")





