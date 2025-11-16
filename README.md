# NLU-Project

Pour entrainer transformer2 et GRU :
    Executez d'abord "fastText_training.py" (avec Python 3.10). FastText training importe la tokenisation fastText.
    Ensuite exécutez IntentAndSlotDataSet (dataset_extract est obsolète). Classifie nos intent et fait le slot filling

Une fois dans un de ces deux dossier (transformer2 ou GRU), exécutez le modèle (fichier se terminant par Model). Puis entrainez le (exécuter le fichier commençant par train) et évaluez le (exécuter le fichier commençant par test).
