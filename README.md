# La polarisation politique en Europe (ESS11)

Projet de Python pour la data science — ENSAE, 2026.
Auteurs : Maxence Menu, Gaël Le Guest, Samuel Placek.

## Sujet

À quoi la position gauche-droite et l'extrémité politique des Européens sont-elles liées, et ces liens varient-ils selon les pays et selon l'âge ?

Nous utilisons la 11e vague de l'*European Social Survey* (ESS11, 2023-2024, 28 pays européens), enrichie avec des données sur les partis votés (PartyFacts, ParlGov) et sur les pays (API Eurostat). Nous reproduisons des figures de trois articles : DellaPosta, Shi et Macy (2015), Baldassarri et Gelman (2008) et Kozlowski et Murphy (2021).

## Lancer le projet

Le projet a été exécuté sur le SSP Cloud (service VSCode-python).

```bash
git clone https://github.com/samuel-placek/polarisation-ess11.git
cd polarisation-ess11
```

Ouvrir ensuite `rapport.ipynb` et tout exécuter (Run All, environ 2 minutes). La première cellule installe les packages de `requirements.txt`.

**Données.** Les fichiers sont trop lourds pour GitHub. Le notebook les télécharge automatiquement dans `data/` depuis le bucket public `https://minio.lab.sspcloud.fr/gleguestensae/`. En cas d'échec, il suffit de placer les fichiers à la main :

- `data/ESS11e04_2.csv` : fichier intégré ESS11, édition 4.2 ([portail ESS](https://ess.sikt.no/en/datafile/242aaa39-3bbb-40f5-98bf-bfb1ce53d8ef)) ;
- `data/externes/essprt-all.csv` : partis de l'ESS dans PartyFacts ;
- `data/externes/partyfacts-external-parties.csv` : liens entre PartyFacts et les autres bases de partis ;
- `data/externes/view_party.csv` : partis de ParlGov.

## Organisation du dépôt

```
rapport.ipynb        notebook final : introduction, parties 1 à 3, conclusion
scripts/
    donnees.py       partie 1 : récupération, nettoyage et enrichissement des données
    correlations.py  partie 2 : corrélations entre opinions, position et extrémité
    alignement.py    partie 3 : simulation de Baldassarri et Gelman, alignement selon l'âge
    outils.py        fonctions communes (moyenne et corrélation pondérées)
figures/             les 9 figures du rapport
data/                données (non versionnées, remplies par le notebook)
requirements.txt     packages Python nécessaires
```

## Contenu

1. **Partie 1 — Données.** Codes de non-réponse, valeurs manquantes par pays, recodage sur [0, 1], variables de polarisation (`lrscale`, `lrscaleabs`), indice de politisation, indices thématiques, enrichissements PartyFacts/ParlGov et Eurostat.
2. **Partie 2 — Corrélations.** Corrélations pondérées pays par pays entre 49 opinions, la position et l'extrémité ; reproduction des figures de DellaPosta et al. et de Baldassarri et Gelman.
3. **Partie 3 — Alignement et âge.** Simulation du mécanisme de Baldassarri et Gelman ; alignement des opinions sur l'axe gauche-droite selon l'âge (adaptation de Kozlowski et Murphy) ; tests de robustesse.

## Sources

- ESS ERIC (2024). *European Social Survey Round 11*, édition 4.2. Licence CC BY-NC-SA 4.0.
- PartyFacts : https://partyfacts.herokuapp.com
- ParlGov : https://www.parlgov.org
- Eurostat : indicateurs `tec00114`, `ilc_di12`, `une_rt_a`
