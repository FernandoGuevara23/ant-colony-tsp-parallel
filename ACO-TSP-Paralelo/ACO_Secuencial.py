import numpy as np
from time import time
import matplotlib.pyplot as plt


def load_tsp_space(path):
    coords = []
    with open(path, "r") as f:
        for line in f:
            if line.strip().upper().startswith("NODE_COORD_SECTION"):
                break
        for line in f:
            s = line.strip()
            if s == "" or s.upper().startswith("EOF"):
                break
            parts = s.split()
            if len(parts) >= 3:
                coords.append([float(parts[1]), float(parts[2])])
    return np.array(coords, dtype=float)



def compute_distances(space):
    diff = space[:, None, :] - space[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    return np.rint(dist).astype(int)



def formulaprob(alpha, beta, eta, pher, visitadas, nodo):
    nodos = np.where(~visitadas)[0]
    vis = eta[nodo, nodos]
    phe = pher[nodo, nodos]
    num = (phe**alpha) * (vis**beta)
    prob = num / np.sum(num)
    return nodos, prob



def actualizacion_local(pher, i, j, xi, tau0):
    tau = (1 - xi) * pher[i, j] + xi * tau0
    pher[i, j] = tau
    pher[j, i] = tau

def actualizacion_global(pher, ruta, costo, phi):
    pher *= (1 - phi)
    inc = phi * (1.0 / costo)
    for a, b in zip(ruta, ruta[1:] + [ruta[0]]):
        pher[a, b] += inc
        pher[b, a] += inc



space = load_tsp_space(
    "C:/Users/aleja/OneDrive/Documentos/Computo Avanzado/TSP-PACO/kroA100.tsp"
)
N = space.shape[0]

dist = compute_distances(space)

eta = np.zeros_like(dist, dtype=float)
mask = dist > 0
eta[mask] = 1.0 / dist[mask]

alpha = 1
beta = 5
xi = 0.1
phi = 0.01
n_hormigas = 80
iteraciones =2000
tau0 = 0.1
corridas = 33             

tiempos = []
costos = []

mejor_ruta_global_total = None
mejor_costo_global_total = float("inf")

for c in range(corridas):
    print(f"Corrida {c+1}/{corridas}")

    pher = np.full((N, N), tau0)
    mejor_global = float("inf")

    start = time()

    for it in range(iteraciones):

        mejor_iter = float("inf")
        mejor_ruta_iter = None

        for _ in range(n_hormigas):

            visitadas = np.zeros(N, dtype=bool)
            nodo = np.random.randint(0, N)
            visitadas[nodo] = True
            ruta = [nodo]

            while not np.all(visitadas):
                nodos_disp, prob = formulaprob(alpha, beta, eta, pher, visitadas, nodo)
                siguiente = np.random.choice(nodos_disp, p=prob)

                actualizacion_local(pher, nodo, siguiente, xi, tau0)

                ruta.append(siguiente)
                visitadas[siguiente] = True
                nodo = siguiente

            costo = sum(dist[ruta[i], ruta[(i+1) % N]] for i in range(N))

            if costo < mejor_iter:
                mejor_iter = costo
                mejor_ruta_iter = ruta

        actualizacion_global(pher, mejor_ruta_iter, mejor_iter, phi)

        
        if mejor_iter < mejor_global:
            mejor_global = mejor_iter

            if mejor_global < mejor_costo_global_total:
                mejor_costo_global_total = mejor_global
                mejor_ruta_global_total = mejor_ruta_iter
        

        if it % 50 == 0:
            print(f"Iter {it} | mejor costo global: {mejor_global}")

    dur = time() - start
    tiempos.append(dur)
    costos.append(mejor_global)

    print(f"Tiempo corrida {c+1}: {dur:.2f}s")
    print(f"Mejor costo corrida {c+1}: {mejor_global}")


np.save("tiempos_ACO_Secuencial2000i.npy", np.array(tiempos))
np.save("costos_ACO_Secuencial2000i.npy", np.array(costos))
np.save("mejor_ruta_global.npy2000i", np.array(mejor_ruta_global_total))

print("\nTiempos guardados en tiempos_ACO_Secuencial.npy")
print("Costos guardados en costos_ACO_Secuencial.npy")

print(mejor_ruta_global_total)
print("Costo:", mejor_costo_global_total)

plt.figure(figsize=(8,5))
plt.boxplot([tiempos], labels=["Secuencial"])
plt.title("Tiempos ACO Secuencial (33 corridas)")
plt.xlabel("Modo")
plt.ylabel("Tiempo (s)")
plt.savefig("boxplot_tiempos_ACO_secuencial2000i.png", dpi=300, bbox_inches="tight")
plt.show()
