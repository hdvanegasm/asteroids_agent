import matplotlib.pyplot as plt
import numpy as np

x = np.array(["Asteriods", "Asteroids - Sin aleatoriedad", "Asteroids - C modificado"])
y = np.array([411.1, 686.5, 867.55])
e = np.array([196.08, 232.82, 353.49])

plt.errorbar(x, y, e, linestyle='None', marker='o')
plt.title("Puntaje y su desviación estándar para Asteroids")
plt.xlabel("Agente")
plt.ylabel("Puntaje")

plt.show()