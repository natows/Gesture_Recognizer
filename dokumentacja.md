„Sterowanie komputerem przy pomocy gestów dłoni w czasie rzeczywistym – porównanie klasyfikatorów ML i prototyp aplikacji użytkowej”

Etap pierwszy
Wykorzystanie media pipe do ekstrakcji cech z obrazu - 21 wspolrzednych i label gestu
Zebrano 4 następujące gesty - palm, I,V oraz ok
Utworzenie prostego modelu kNN do klasyfikacji danych z pliku csv - model radzi sobie z danymi
bardzo dobrze, przy danych początkowych accuracy wynosiło 100 % na 10 prób. Przy dodaniu większej ilości danych
modelowi zdarza się pomylić 1/2 gesty. Przy sprawdzeniu metodą cross-validation accuracy wynosi 98%. 
Cross-validation pokazuje dobre wyniki we wszystkich foldach, bez dużych wahań: Wyniki cross-validation: [0.99326599 0.98989899 0.95608108 0.98648649 0.99662162].
Wskazuje to na brak przeuczenia a poprostu dobrą skuteczność modelu.
Tak wysoka dokładność może wynikać z czystości danych. Być może trzeba to będzie poprawić.