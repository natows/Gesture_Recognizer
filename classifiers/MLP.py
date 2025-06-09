from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle


import pandas as pd

df = pd.read_csv("../data_sources/gestures.csv")  
X = df.drop("label", axis=1).values
y = df["label"].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
mlp.fit(X_train, y_train)


y_pred = mlp.predict(X_test)
print("Accuracy:", mlp.score(X_test, y_test))
print("Cross-validation scores:", cross_val_score(mlp, X, y, cv=5))
print("Classification Report:")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred, labels=mlp.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title("Macierz pomyłek")
plt.savefig("mlp_confusion_matrix.png")
plt.show()


with open('../models/MLP/MLP.pkl', 'wb') as f:
    pickle.dump(mlp, f)
print("Model zapisany do 'model.pkl'")



# Accuracy: 1.0
# Cross-validation scores: [0.98058252 1.         0.99675325 1.         0.99675325]
# Classification Report:
#               precision    recall  f1-score   support

#            L       1.00      1.00      1.00       140
#            V       1.00      1.00      1.00       131
#           ok       1.00      1.00      1.00        51
#         palm       1.00      1.00      1.00       141

#     accuracy                           1.00       463
#    macro avg       1.00      1.00      1.00       463
# weighted avg       1.00      1.00      1.00       463
