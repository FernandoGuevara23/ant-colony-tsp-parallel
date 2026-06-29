from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from time import time


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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

if rank == 0:
    space = load_tsp_space("/home/admin/MPI_DEBIAN_WSL/kroA100.tsp")
    N = space.shape[0]

    diff = space[:, None, :] - space[None, :, :]
    dist = np.rint(np.sqrt(np.sum(diff**2, axis=2))).astype(int)

    eta = np.zeros_like(dist, dtype=float)
    mask = dist > 0
    eta[mask] = 1.0 / dist[mask]
else:
    N = None
    dist = None
    eta = None

N = comm.bcast(N, root=0)
dist = comm.bcast(dist, root=0)
eta = comm.bcast(eta, root=0)

def formulaprob(alpha, beta, eta, pher, visitadas, nodo):
    nodos = np.where(~visitadas)[0]

    if len(nodos) == 0:
        return [], None

    vis = eta[nodo, nodos]
    phe = pher[nodo, nodos]

    num = (phe**alpha) * (vis**beta)
    s = np.sum(num)

    if s == 0 or np.isnan(s):
        prob = np.full(len(nodos), 1/len(nodos))
        return nodos, prob

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


alpha = 1
beta = 5
xi = 0.1
rho = 0.1
n_hormigas = 80
iteraciones = 1000
tau0 = 0.1
corridas = 33

resultados = []
mejor_costo_global = None
mejor_ruta_global = None


for corrida in range(corridas):

    if rank == 0:
        inicio = time()

    pher = np.full((N, N), tau0) if rank == 0 else None
    pher = comm.bcast(pher, root=0)

    mejor_global = float("inf")
    mejor_ruta_global = None

    hormigas_local = n_hormigas // size

    if rank == 0:
        evolucion_costos = []

    if rank == 0:
        print(f"\nCorrida {corrida+1}/{corridas}")
        print(f"Ejecutando con {size} procesos, {hormigas_local} hormigas por proceso.")

    for it in range(iteraciones):

        mejor_local = float("inf")
        mejor_ruta_local = None

        pher_local = pher.copy()
        for _ in range(hormigas_local):
            visitadas = np.zeros(N, dtype=bool)
            nodo = np.random.randint(0, N)
            visitadas[nodo] = True
            ruta = [nodo]

            while not np.all(visitadas):
                nodos_disp, prob = formulaprob(alpha, beta, eta, pher_local, visitadas, nodo)
                siguiente = np.random.choice(nodos_disp, p=prob)

                actualizacion_local(pher_local, nodo, siguiente, xi, tau0)

                ruta.append(siguiente)
                visitadas[siguiente] = True
                nodo = siguiente

            costo = sum(dist[ruta[i], ruta[(i + 1) % N]] for i in range(N))

            if costo < mejor_local:
                mejor_local = costo
                mejor_ruta_local = ruta

        best_costs = comm.gather(mejor_local, root=0)
        best_routes = comm.gather(mejor_ruta_local, root=0)

        if rank == 0:

            idx = np.argmin(best_costs)
            mejor_iter = best_costs[idx]
            mejor_ruta_iter = best_routes[idx]

            actualizacion_global(pher, mejor_ruta_iter, mejor_iter, rho)

            
            if mejor_iter < mejor_global:
                mejor_global = mejor_iter
                mejor_ruta_global = mejor_ruta_iter

            evolucion_costos.append(mejor_global)

            if it % 50 == 0:
                print(f"[Iter {it}] Mejor costo global: {mejor_global}")

        else:
            mejor_iter = None
            mejor_ruta_iter = None

        pher = comm.bcast(pher, root=0)

    if rank == 0:
        duracion = time() - inicio
        resultados.append(duracion)
        print(f"Tiempo corrida {corrida+1}: {duracion:.2f} s")

    if mejor_costo_global is None or mejor_global < mejor_costo_global:
        mejor_costo_global = mejor_global
        mejor_ruta_global = mejor_ruta_global


if rank == 0:
    print("Mejor costo:", mejor_costo_global)
    print("Mejor ruta:", [int(x) for x in mejor_ruta_global])

    np.save("MPI_final_pher2.npy", pher)
    np.save("MPI_final_ruta2.npy", np.array(mejor_ruta_global))
    np.save("MPI_final_costo2.npy", np.array([mejor_costo_global]))
    np.save("MPI_final_tiempos2.npy", np.array(resultados))

    plt.boxplot(resultados, tick_labels=["MPI"])
    plt.title(f"Tiempos MPI ACO ({corridas} corridas)")
    plt.ylabel("Tiempo (s)")
    plt.savefig("MPI_final_Imagen2.png", dpi=300, bbox_inches="tight")
    print("Gráfica guardada como tiempos_MPI_ACO_CORREGIDO.png")
