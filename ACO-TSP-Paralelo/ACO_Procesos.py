import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from time import time

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
    s = np.sum(num)

    if s <= 0 or np.isnan(s):
        prob = np.ones_like(num) / len(num)
    else:
        prob = num / s

    return nodos, prob

def actualizacion_local(pher, i, j, xi, tau0):
    tau_nuevo = (1 - xi) * pher[i, j] + xi * tau0
    pher[i, j] = tau_nuevo
    pher[j, i] = tau_nuevo

def actualizacion_global(pher, ruta, costo, rho):
    pher *= (1 - rho)
    inc = rho * (1.0 / costo)
    for a, b in zip(ruta, ruta[1:] + [ruta[0]]):
        pher[a, b] += inc
        pher[b, a] += inc

def worker_aco(i, numero_procesos, hormigas_totales, pher, dist, eta, alpha, beta, xi, tau0):
    hormigas_local = hormigas_totales // numero_procesos

    N = pher.shape[0]
    best_cost = float("inf")
    best_route = None

    for _ in range(hormigas_local):

        visitadas = np.zeros(N, dtype=bool)
        nodo = np.random.randint(0, N)
        visitadas[nodo] = True
        ruta = [nodo]

        while not np.all(visitadas):
            nodos_disp, prob = formulaprob(alpha, beta, eta, pher, visitadas, nodo)
            siguiente = np.random.choice(nodos_disp, p=prob)
            ruta.append(siguiente)

            actualizacion_local(pher, nodo, siguiente, xi, tau0)

            visitadas[siguiente] = True
            nodo = siguiente

        costo = sum(dist[ruta[i], ruta[(i + 1) % N]] for i in range(N))

        if costo < best_cost:
            best_cost = costo
            best_route = ruta

    return best_route, best_cost

if __name__ == "__main__":

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
    rho = 0.1
    iteraciones = 100
    tau0 = 0.1
    xi = 0.1

    hormigas_totales = 80

    procesos = [8]
    corridas = 33
    resultados = []

    mejor_ruta_global = None
    mejor_costo_global = float("inf")

    for proc in procesos:

        tiempos = []

        for corrida in range(corridas):
            print(corrida)

            pher = np.full((N, N), tau0)

            st = time()
            pool = mp.Pool(proc)

            mejor_global = float("inf")
            mejor_ruta = None

            for it in range(iteraciones):

                results = [
                    pool.apply_async(
                        worker_aco,
                        (i, proc, hormigas_totales, pher, dist, eta, alpha, beta, xi, tau0)
                    )
                    for i in range(proc)
                ]

                results = [r.get() for r in results]

                ruta_best, costo_best = min(results, key=lambda x: x[1])
                actualizacion_global(pher, ruta_best, costo_best, rho)

                if costo_best < mejor_global:
                    mejor_global = costo_best
                    mejor_ruta = ruta_best

                if it % 50 == 0:
                    print(f"Iter {it}: costo={mejor_global}")

            pool.close()
            tiempos.append(time() - st)


            if mejor_global < mejor_costo_global:
                mejor_costo_global = mejor_global
                mejor_ruta_global = mejor_ruta

        resultados.append(tiempos[:])
    
    
    print(mejor_costo_global)
    print(mejor_ruta_global)
    
    np.save("mejor_ruta_global_ACO_p_phi0.01.npy", np.array(mejor_ruta_global, dtype=int))
    np.save("mejor_costo_global__phi0.01.npy", np.array([mejor_costo_global]))

    np.save("tiempos_procesos_phi0.01.npy", np.array(resultados, dtype=object))

    plt.boxplot(resultados, labels=[str(p) for p in procesos])
    plt.title("Tiempos ACO por número de procesos (33 corridas)")
    plt.xlabel("Procesos")
    plt.ylabel("Tiempo (s)")
    plt.savefig("boxplot_tiempos_ACO_Process__phi0.01.png", dpi=300, bbox_inches="tight")
    plt.show()
