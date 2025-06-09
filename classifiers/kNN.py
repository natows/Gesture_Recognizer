import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle


df = pd.read_csv('../data_sources/gestures.csv')

X = df.drop('label', axis=1)
y = df['label']

#nie musza byc skalowane bo juz sa w zakresie 0,1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title("Macierz pomyłek")
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


#model knn ma w wiekszosci przypadkow 100% accuracy na zbiorze, przy dodaniu paru sztucznych danych, model się myli
#wiec rzeczywiscie dziala jak powinien

#sprawdzenie czy model jest stabilny i nie overfitowany
from sklearn.model_selection import cross_val_score
import numpy as np


scores = cross_val_score(knn, X, y, cv=5)
#dzielisz dane na k foldow(czesci) i trenijesz model k razy 
# 1 iteracja trenujesz na 2,3,4,5 foldzie a testujesz na 1
# 2 iteracja trenujesz na 1,3,4,5 foldzie a testujesz na 2 


print("Wyniki cross-validation:", scores)
print("Średnia dokładność:", np.mean(scores))


with open('models/kNN/kNN.pkl', 'wb') as f:
    pickle.dump(knn, f)
print("Model zapisany do 'model.pkl'")



# Accuracy: 0.9955056179775281
# Wyniki cross-validation: [0.99326599 0.98989899 0.95608108 0.98648649 0.99662162]
# Średnia dokładność: 0.9844708344708344


