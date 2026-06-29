# ACO-TSP Paralelo

Implementación del algoritmo **Ant Colony Optimization (ACO)** aplicado al **Problema del Viajante (TSP)**, con tres versiones: secuencial, paralelización por procesos (multiprocessing) y distribuida con MPI.

**Materia:** Cómputo Avanzado  
**Alumno:** Fernando Alejandro Guevara Rodriguez — NUA: 283803  
**Docente:** Marco Aurelio Sotelo Figueroa  
**Institución:** Universidad de Guanajuato, Campus Guanajuato — División de Ciencias Económico Administrativas

---

## Descripción

El algoritmo ACO simula el comportamiento de hormigas que depositan feromonas para encontrar rutas cortas. Aplicado al TSP sobre la instancia **kroA100** (100 nodos de TSPLIB), se compararon tres estrategias de implementación bajo 33 corridas cada una.

### Parámetros del algoritmo

| Parámetro | Valor |
|-----------|-------|
| α (alpha) | 1 |
| β (beta) | 5 |
| ξ (xi) — evaporación local | 0.1 |
| ρ (rho) — evaporación global | 0.1 |
| Número de hormigas | 80 |
| Iteraciones | 1000 |
| τ₀ (tau0) | 0.1 |
| Corridas | 33 |
| Procesos paralelos | 8 |

---

## Estructura del proyecto

```
ACO-TSP-Paralelo/
├── ACO_Secuencial.py                        # Implementación secuencial
├── ACO_Procesos.py                          # Paralelización con multiprocessing
├── ACO_MPI.py                               # Paralelización distribuida con MPI
├── kroA100.tsp                              # Instancia TSP (TSPLIB)
├── mejor_ruta_global_ACO_p_phi0_01.npy     # Mejor ruta encontrada (Procesos)
├── mejor_costo_global__phi0_01.npy         # Mejor costo encontrado (Procesos)
├── tiempos_procesos_phi0_01.npy            # Tiempos por corrida (Procesos)
├── boxplot_tiempos_ACO_Process__phi0_01.png # Gráfica de tiempos
├── Reporte_ACO_Tiempos.pdf                  # Reporte de tiempos (boxplot)
└── Reporte_Implementacion_ACO.pdf           # Reporte de implementación
```

---

## Implementaciones

### 1. Secuencial (`ACO_Secuencial.py`)
Versión base. Ejecuta todas las hormigas en un solo proceso. Sirve como referencia para medir el speedup de las versiones paralelas.

- **Mejor costo encontrado:** 23,869

### 2. Procesos — `multiprocessing` (`ACO_Procesos.py`)
Divide las hormigas entre 8 procesos usando `mp.Pool`. El proceso principal recolecta los resultados y aplica la actualización global de feromonas.

- **Mejor costo encontrado:** 21,594

### 3. MPI (`ACO_MPI.py`)
Distribuye las hormigas entre 8 procesos MPI. El proceso raíz (rank 0) centraliza los resultados, actualiza feromonas y redistribuye la matriz a todos los procesos.

- **Mejor costo encontrado:** 21,318

---

## Resultados de tiempos (33 corridas, 8 procesos)

| Implementación | Tiempo mediano aprox. |
|----------------|-----------------------|
| Secuencial     | ~320 s                |
| Procesos       | ~9.7 s                |
| MPI            | ~76 s                 |

La versión con `multiprocessing` resultó la más rápida en este entorno (Windows/WSL), mientras que MPI introdujo overhead de comunicación pero sigue siendo considerablemente más rápida que la versión secuencial.

---

## Requisitos

```bash
pip install numpy matplotlib mpi4py
```

Para la versión MPI es necesario tener instalado un proveedor de MPI (como OpenMPI o MPICH).

---

## Ejecución

### Secuencial
```bash
python ACO_Secuencial.py
```

### Procesos
```bash
python ACO_Procesos.py
```

### MPI (con N procesos)
```bash
mpiexec -n 8 python ACO_MPI.py
```

> **Nota:** Ajusta la ruta del archivo `.tsp` dentro de cada script antes de ejecutar.

---

## Referencias

- mastqe. (s.f.). *TSPLIB solutions repository*. GitHub. https://github.com/mastqe/tsplib/blob/master/solutions
- Dorigo, M., & Stützle, T. (2004). *Ant Colony Optimization*. MIT Press.
