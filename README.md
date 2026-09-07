== Balayage axe 1 : niveau de bruit de mesure ==
   jeu d'évaluation : dev (graine de base 20250)
   niveaux          : [-20, -15, -10, -5, 0, 5, 10, 15, 20, 30]
   trajectoires     : 10 par niveau
   offset figé      : 0.3
   modèles          : baselines étroites, aucun réentraînement

   checkpoint chargé : graine 42
   checkpoint chargé : graine 1234
   checkpoint chargé : graine 7

   valeur=   -20 | Delta_dB moyen =  +1.31 dB | 13 s
   valeur=   -15 | Delta_dB moyen =  -1.00 dB | 13 s
   valeur=   -10 | Delta_dB moyen =  -4.06 dB | 13 s
   valeur=    -5 | Delta_dB moyen =  -6.59 dB | 13 s
   valeur=     0 | Delta_dB moyen =  -6.25 dB | 12 s
   valeur=     5 | Delta_dB moyen =  -5.25 dB | 13 s
   valeur=    10 | Delta_dB moyen =  -6.27 dB | 13 s
   valeur=    15 | Delta_dB moyen =  -7.91 dB | 13 s
   valeur=    20 | Delta_dB moyen =  -4.64 dB | 13 s
   valeur=    30 | Delta_dB moyen =  -1.48 dB | 12 s

== Repères de franchissement ==
   parité (0 dB)    | courbe moyenne : -17.17
      graine 7      : -16.21
      graine 42     : -18.48
      graine 1234   : -16.48
   seuil +3 dB      | courbe moyenne : hors plage balayée
      graine 7      : hors plage
      graine 42     : hors plage
      graine 1234   : hors plage

== Résultats -> ./runs/balayage_bruit/balayage_bruit_dev.json
== Figure    -> ./runs/balayage_bruit/balayage_bruit_dev.png
