<img width="1820" height="700" alt="features" src="https://github.com/user-attachments/assets/022daf33-206e-4ae6-a5ba-351b63279b9e" />
<img width="1820" height="1260" alt="dynamique" src="https://github.com/user-attachments/assets/6bc8048e-eecd-4d54-9e9e-2c6b5992dfae" />



= Attribution : axe offset initial ==
   échelles     : [0.3, 1.0, 2.0]
   trajectoires : 20 par échelle
   modèle       : baseline étroite, graine 42

   collecte offset 0.3 ...
   collecte offset 1.0 ...
   collecte offset 2.0 ...

== Dynamique (médiane sur les 20 premiers pas) ==
  offset       ||dy||       ||KG||   depassement      MSE pos
     0.3        0.090        1.332         0.050       0.1936
     1.0        0.103        1.500         0.019       1.5687
     2.0        0.144        1.541         0.013       6.1930

== Features (moyenne sur toute la trajectoire) ==
  offset        norme   cosinus vs ref
     0.3       1.0741           1.0000
     1.0       1.0656           0.8425
     2.0       0.9663           0.7886

Lecture : si la norme reste plate à 1 et que seul le cosinus
décroche, le réseau est bien aveugle à l'ampleur de son erreur.

== Résumé   -> ./runs/attribution_offset/attribution_offset.json
== Figure 1 -> ./runs/attribution_offset/dynamique.png
== Figure 2 -> ./runs/attribution_offset/features.png
<img width="1820" height="1260" alt="dynamique" src="https://github.com/user-attachments/assets/85be4086-b5b5-4caa-ab44-8028e7fbe0a6" />
