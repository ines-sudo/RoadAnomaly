# Noussa — démarrage DeepLabV3+

Cette livraison ajoute ta partie au dépôt. Les données RoadAnomaly restent inchangées.
Le code et le notebook sont en anglais conformément au dépôt; ce guide t'accompagne en français.

## Démarrer
1. Extraire le ZIP complet.
2. Ouvrir un terminal dans RoadAnomaly-main.
3. Utiliser Python 3.12 et installer `pip install -r requirements/dl.txt`.
4. Les données CamVid et leur manifeste sont déjà inclus. La commande `python -m dl.data.prepare_camvid` sert uniquement à les reconstruire si nécessaire.
5. Ouvrir `dl/notebooks/02_deeplabv3plus.ipynb` dans Jupyter/VS Code.
6. Pour le premier essai réel : `python -m dl.training.ines_camvid --epochs 3`.

Un GPU est préférable. Aucun entraînement complet n'est présenté comme déjà réalisé.
Le premier téléchargement des poids ImageNet exige Internet.
Les fichiers de sortie sont dans `artifacts/ines-deeplabv3plus/`.
CamVid fournit les classes habituelles : ciel, bâtiment, route, voiture, piéton, etc.
Les exemples sont répartis par vidéo entière, pas aléatoirement par image.
Le modèle apprend ces classes; 1-max(softmax) fournit ensuite un score d'incertitude.
Ce score ne garantit pas la détection d'un obstacle.

## Ta contribution
Comprendre et présenter encodeur, convolutions dilatées, ASPP, fusion bas niveau et décodeur.
Avec Emna : entraîner, inspecter les erreurs, comparer encodeur gelé et fine-tuning.
Seg-Grad-CAM est disponible dans dl/evaluation/xai.py; ses tests de fidélité et les métriques finales restent à compléter.

## Limites actuelles
Livraison de démarrage, profil learning. Pas encore de performance officielle, XAI final,
seuil calibré ou promotion de modèle. Le corpus CamVid urbain n'est pas certifié sans anomalie;
une revue est nécessaire avant un protocole officiel. Conserver les conditions et citations
CamVid/SegNet pour un usage académique. Ne pas publier des scores de données synthétiques.

## Validation de la livraison
Ruff et mypy passent. Les tests du modèle vérifient gradients, export, rechargement et attributions.
La suite complète du dépôt ZIP rencontre des erreurs dans l’audit existant en l’absence de métadonnées Git; ces tests ne sont pas modifiés.
Aucun entraînement complet, score final ou résultat officiel n’est annoncé.
