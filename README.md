# Régression logistique One-vs-Rest
Ce travail fournis l'approche One-vs-Rest de la régression logistique
<h1>Préalable</h1>
Nous utilisons les bibliothèque numpy et pandas pour la gestion des données


Pour afficher les courbes, nous avons utilisé matplotlib qui n'est obligatoire que pour visualiser les résultats que nous avons utilisés pour rédiger notre rapport. 

Vous pouvez changer les hyperparamètres learning_rate, num_iterations de la classe MultiClassLogisticRegression
<h1>Utilisation</h1>

   1. Modifier le chemin d'accès vers les fichiers csv, train.csv et test.csv

   
      train_data = pd.read_csv('/chemin/vers/train.csv')


      test_data = pd.read_csv('/chemin/vers/test.csv')


   2. En excécutant le code, celui ci effectuera l'entraînement sur l'ensemble d'entraînement et la prédiction sur l'ensemble de test

   
   3. Les prédictions faites par le modèles seront sauvegardées dans un fichier CSV, sample_submission_logreg.csv
<h1>Fonctionalités</h1>
   1. Lecture et sauvegarde des données d'entrainement


   2. Lecteur et sauvegarde des données de test

   
   3. Supression des colonnes SNo et Label dans l'ensemble d'entrainement

   
   4. Suppression de la colonne SNo dans l'ensemble de test

   
   5. Normalisation des données d'entrainement et de test 

   
   6. La régression logistique binaire est prise en compte par la classe LogisticRegression

    
   7. La régression logistique multi-classe One-vs-Rest est prise en compte par la classe MultiClassLogisticRegression

    
   8. Entrainement du modèle sur le train (l'ensemble d'entrainement)

    
   9. Prédiction du modèle sur le test (l'ensemble de test)

    
   10. Sauvegarde des données dans le fichier sample_submission_logreg.csv
