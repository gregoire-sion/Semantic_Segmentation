import math

def intersection_cercles(x1, y1, r1, x2, y2, r2):
    # Calcul de la distance entre les centres
    d = math.hypot(x2 - x1, y2 - y1)

    # Vérification des cas où il n'y a pas d'intersection
    if d > r1 + r2:
        return []  # Les cercles sont trop éloignés
    if d < abs(r1 - r2):
        return []  # Un cercle est contenu dans l'autre
    if d == 0 and r1 == r2:
        return []  # Les cercles sont confondus (infinité de points)

    # Calcul des distances vers le point central d'intersection
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = math.sqrt(abs(r1**2 - a**2))

    # Point P2 (point sur la ligne entre les deux centres)
    x3 = x1 + a * (x2 - x1) / d
    y3 = y1 + a * (y2 - y1) / d

    # Si les cercles se touchent en un seul point (tangents)
    if d == r1 + r2 or d == abs(r1 - r2):
        return [[x3, y3]]

    # Calcul des deux points d'intersection P3 et P4
    rx = -h * (y2 - y1) / d
    ry = h * (x2 - x1) / d

    intersection1 = [x3 + rx, y3 + ry]
    intersection2 = [x3 - rx, y3 - ry]

    return [intersection1, intersection2]

# --- Exemple d'utilisation ---
points = intersection_cercles(0, 0, 5, 4, 0, 3)
print(points)
