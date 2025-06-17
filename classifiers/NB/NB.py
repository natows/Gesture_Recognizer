import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
import os  

df = pd.read_csv("../../data_sources/gestures.csv")


wybrane_gesty = ['V', 'palm', 'ok', 'L']

df_filtered = df[df['label'].isin(wybrane_gesty)]

X = df_filtered.drop("label", axis=1).values
y = df_filtered["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) 

clf = GaussianNB()
clf.fit(X_train, y_train)

print("Accuracy:", clf.score(X_test, y_test))
print("Cross-validation scores:", cross_val_score(clf, X, y, cv=5))
print("Classification Report:")
print(classification_report(y_test, clf.predict(X_test)))

cm = confusion_matrix(y_test, clf.predict(X_test), labels=wybrane_gesty)  
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=wybrane_gesty)  
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title("Macierz pomyłek Naive Bayes")
plt.savefig("nb_confusion_matrix.png")
plt.show()

os.makedirs('../../models/NB', exist_ok=True)
with open('../../models/NB/NaiveBayes_model.pkl', 'wb') as f: 
    pickle.dump(clf, f)
print("Model zapisany do '../../models/NB/NaiveBayes_model.pkl'") 