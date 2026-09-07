== Balayage axe 2 : amplitude de l'offset initial ==
   jeu d'évaluation : dev (graine de base 21250)
   échelles         : [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0]
   trajectoires     : 150 par échelle
   bruit figé       : 0.0 dB (nominal)
   EKF de référence : nominal (pas d'oracle sur cet axe)

   checkpoint chargé : graine 42
   checkpoint chargé : graine 1234
   checkpoint chargé : graine 7

   valeur=   0.0 | Delta_dB moyen =  -4.30 dB | 177 s
   valeur=   0.3 | Delta_dB moyen =  -6.25 dB | 187 s
   valeur=   0.6 | Delta_dB moyen =  -1.22 dB | 177 s
   valeur=   1.0 | Delta_dB moyen =  +2.47 dB | 177 s
   valeur=   1.5 | Delta_dB moyen = +10.26 dB | 185 s
   valeur=   2.0 | Delta_dB moyen = +23.84 dB | 184 s
   valeur=   3.0 | Delta_dB moyen = +29.04 dB | 175 s
   valeur=   5.0 | Delta_dB moyen = +38.33 dB | 173 s

== Repères de franchissement ==
   parité (0 dB)    | courbe moyenne : +0.73
      graine 7      : +0.79
      graine 42     : +0.71
      graine 1234   : +0.70
   seuil +3 dB      | courbe moyenne : +1.03
      graine 7      : +1.08
      graine 42     : +1.03
      graine 1234   : +1.00

== Résultats -> ./runs/balayage_offset/balayage_offset_dev.json
== Figure    -> ./runs/balayage_offset/balayage_offset_dev.png
