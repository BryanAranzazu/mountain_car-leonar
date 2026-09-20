# Lista de reproducibilidad

Adaptación de la lista de verificación de reproducibilidad en aprendizaje automático de Pineau et al. (2021). Cada punto señala dónde está la evidencia en este repositorio.

## Algoritmos

| Punto | Estado | Dónde |
|---|---|---|
| Descripción clara del planteamiento matemático y del algoritmo | Sí | README, secciones 1 y 3 |
| Supuestos explícitos | Sí | README 3.1 (uso de `terminated`), 3.3 (naturaleza off-policy de la corrección) |
| Análisis de complejidad o de costo | Parcial | README 7: pasos de entorno y tiempo de reloj; sin análisis asintótico |

## Código

| Punto | Estado | Dónde |
|---|---|---|
| Dependencias especificadas con versión exacta | Sí | `uv.lock`, `.python-version` (Python 3.11) |
| Código de entrenamiento | Sí | `src/mountain_car/agents/`, `scripts/experimento.py` |
| Código de evaluación | Sí | `scripts/experimento.py` (función `evaluar`), `mountaincar load --eval` |
| Modelos entrenados | Sí | `saves/`, descritos en `docs/tarjetas_de_modelo.md` |
| Órdenes exactas para reproducir los resultados | Sí | `make reproducir`; README, sección 2 |
| Pruebas automáticas | Sí | `tests/`, ejecutadas en CI (`.github/workflows/ci.yml`) |

## Resultados

| Punto | Estado | Dónde |
|---|---|---|
| Semillas declaradas | Sí | semilla 7 para entrenar, 507 para los controles, 1007 para la evaluación final |
| Número de corridas de evaluación | Sí | 100 episodios por agente |
| Medida de tendencia central y de variación | Sí | media y desviación estándar; mejor y peor episodio |
| Separación entre selección de modelo y evaluación | Sí | el punto de control se elige con los controles; la cifra reportada usa otras semillas |
| Rango de hiperparámetros explorado y criterio de selección | Parcial | solo se ajustó `explore_repeat`, con la tabla de `diagnostico_exploracion.txt`; el resto son los valores del repositorio base |
| Resultados de todas las corridas, no solo de la mejor | Parcial | se reportan el mejor punto de control y el agente final; la variación entre semillas se midió solo para la tabla |
| Registros crudos disponibles | Sí | `resultados/registros/`, `resultados/metricas/` |
| Resultado negativo documentado | Sí | primera versión fallida de la corrección (README 3.3) y DQN sin corrección (`dqn_sin_correccion.txt`) |

## Infraestructura usada en la corrida reportada

CPU Intel Xeon a 2.10 GHz, 2 núcleos, Linux x86_64, sin GPU. Python 3.11, PyTorch 2.13.0, Gymnasium 1.3.0, NumPy 2.4.6. Tiempos: 4.4 min para Q-Learning (compartiendo CPU con otro proceso) y 13.7 min para DQN con `OMP_NUM_THREADS=1`.

## Límites conocidos de la reproducibilidad

La semilla fija hace repetible la secuencia de números aleatorios, pero PyTorch no garantiza resultados idénticos bit a bit entre versiones, sistemas operativos o arquitecturas de CPU. En otra máquina cabe esperar cifras cercanas, no iguales. La evaluación de la CLI original (`load --eval`) no fija semilla y usa 10 episodios, por lo que varía entre ejecuciones.

## Referencia

Pineau, J., Vincent-Lamarre, P., Sinha, K., Larivière, V., Beygelzimer, A., d'Alché-Buc, F., Fox, E., & Larochelle, H. (2021). Improving reproducibility in machine learning research. *Journal of Machine Learning Research, 22*(164), 1-20.
