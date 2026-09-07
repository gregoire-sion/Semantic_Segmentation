<img width="1820" height="700" alt="balayage_bruit_dev" src="https://github.com/user-attachments/assets/e28882ec-e9e6-4040-99cf-8c7a40925a60" />

== Balayage axe 1 : niveau de bruit de mesure ==
   jeu d'évaluation : dev (graine de base 20250)
   niveaux          : [-20, -15, -10, -5, 0, 5, 10, 15, 20, 30]
   trajectoires     : 150 par niveau
   offset figé      : 0.3
   modèles          : baselines étroites, aucun réentraînement

   checkpoint chargé : graine 42
   checkpoint chargé : graine 1234
   checkpoint chargé : graine 7

   valeur=   -20 | Delta_dB moyen =  +1.04 dB | 179 s
   valeur=   -15 | Delta_dB moyen =  -1.69 dB | 174 s
   valeur=   -10 | Delta_dB moyen =  -3.68 dB | 180 s
   valeur=    -5 | Delta_dB moyen =  -4.75 dB | 176 s
   valeur=     0 | Delta_dB moyen =  -6.17 dB | 170 s
   valeur=     5 | Delta_dB moyen =  -5.39 dB | 169 s
   valeur=    10 | Delta_dB moyen =  -5.49 dB | 174 s
   valeur=    15 | Delta_dB moyen =  -5.03 dB | 170 s
   valeur=    20 | Delta_dB moyen =  -3.07 dB | 169 s
   valeur=    30 | Delta_dB moyen =  -0.63 dB | 178 s

== Repères de franchissement ==
   parité (0 dB)    | courbe moyenne : -18.09
      graine 7      : -18.83
      graine 42     : -19.71
      graine 1234   : -16.83
   seuil +3 dB      | courbe moyenne : hors plage balayée
      graine 7      : hors plage
      graine 42     : hors plage
      graine 1234   : hors plage

== Résultats -> ./runs/balayage_bruit/balayage_bruit_dev.json
== Figure    -> ./runs/balayage_bruit/balayage_bruit_dev.png


