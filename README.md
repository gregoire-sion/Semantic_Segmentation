

<img width="1960" height="700" alt="balayage_commande_dev" src="https://github.com/user-attachments/assets/79e0e491-3fa7-4c39-9adc-94ce2a9cef45" />
== Balayage axe 4 : famille de commande ==
   jeu d'évaluation : dev (graine de base 23250)
   familles         : ['nominal_3phases', 'phases3_rand', 'ou', 'ood_3phases', 'ood_brutal']
   entraînement sur : nominal_3phases
   trajectoires     : 150 par famille
   EKF de référence : nominal

   checkpoint chargé : graine 42
   checkpoint chargé : graine 1234
   checkpoint chargé : graine 7

   valeur=nominal_3phases | Delta_dB moyen =  -5.69 dB | 178 s
   valeur=phases3_rand | Delta_dB moyen =  +4.97 dB | 175 s
   valeur=    ou | Delta_dB moyen =  +4.65 dB | 188 s
   valeur=ood_3phases | Delta_dB moyen =  -5.44 dB | 179 s
   valeur=ood_brutal | Delta_dB moyen =  -5.50 dB | 184 s

== Résultats -> ./runs/balayage_commande/balayage_commande_dev.json
== Figure    -> ./runs/balayage_commande/balayage_commande_dev.png
