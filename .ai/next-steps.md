# Prochains petits pas — DSLR Partie B

1. `preprocessing.py` : structure/signatures d'abord, puis implémentation.
   - Point critique : threading des means/stds (calculés au train, réutilisés au predict).
2. Briques math de `logreg_train` testées isolément : sigmoid → cost → gradient.
3. Boucle de descente + one-vs-all + export poids.
4. `logreg_predict.py` + génération `houses.csv`.
5. Tuning lr/itérations jusqu'à ≥ 98 %.
