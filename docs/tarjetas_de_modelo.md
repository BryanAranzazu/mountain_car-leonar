# Tarjetas de modelo

Formato adaptado de Mitchell et al. (2019). Describen los dos agentes entrenados que acompañan al repositorio en `saves/`.

## Agente 1. Q-Learning tabular

| Campo | Valor |
|---|---|
| Archivo | `saves/qlearning_mountaincar.pkl` (18 KB, pickle de Python) |
| Tipo | Tabla Q sobre una rejilla de 20 × 20 celdas, 3 acciones por celda |
| Entrenamiento | 20 000 episodios, semilla 7; se conserva el punto de control del episodio 10 500 |
| Hiperparámetros | `lr=0.1`, `gamma=0.99`, epsilon de 1.0 a 0.01 con factor 0.9995 por episodio |
| Datos | 1.84 millones de pasos de entorno hasta el punto de control; 298 de 400 celdas visitadas |
| Evaluación | 100 episodios voraces, semillas no usadas en entrenamiento ni en selección |
| Resultado | recompensa media -123.0 ± 16.4; bandera en 98 de 100; mejor -110, peor -200 |

Uso previsto. Material docente: muestra el ciclo de Q-learning y los límites de la discretización.

Limitaciones. No supera el umbral convencional de -110. El desempeño oscila durante el entrenamiento: el agente del último episodio evalúa en -153.9, por lo que el archivo guardado corresponde al mejor punto de control y no al final. La celda discretizada no es un estado de Markov. Con otras semillas (1, 2 y 3) el agente final evaluó entre -131.0 y -164.1.

Advertencia de seguridad. El archivo es un pickle. Cargue solo copias obtenidas de este repositorio, porque deserializar un pickle de origen desconocido puede ejecutar código arbitrario.

## Agente 2. DQN con exploración persistente

| Campo | Valor |
|---|---|
| Archivo | `saves/dqn_mountaincar.pt` (211 KB, `torch.save`) |
| Tipo | Perceptrón 2 → 128 → 128 → 3, ReLU, salida lineal; 17 283 parámetros |
| Entrenamiento | 2 500 episodios, semilla 7; se conserva el punto de control del episodio 2 450 |
| Hiperparámetros | Adam `lr=1e-3`, `gamma=0.99`, lote 64, memoria 100 000, red objetivo sincronizada cada 10 episodios, epsilon de 1.0 a 0.01 con factor 0.995, `explore_repeat=0.95` |
| Datos | 368 mil pasos de entorno hasta el punto de control |
| Evaluación | 100 episodios voraces, semillas no usadas en entrenamiento ni en selección |
| Resultado | recompensa media -96.7 ± 7.6; bandera en 100 de 100; mejor -83, peor -104 |

Uso previsto. Material docente: ilustra memoria de repetición, red objetivo y el efecto de la estructura temporal de la exploración.

Limitaciones. Resultado de una sola semilla; falta la evaluación con varias. Durante el entrenamiento aparecen caídas aisladas del desempeño voraz (-175.7 en el episodio 1 850 y -162.1 en el 2 150), coherentes con la ausencia de garantías de convergencia de Q-learning con aproximación no lineal. La entrada no está normalizada. El agente solo es válido para MountainCar-v0 con el límite estándar de 200 pasos.

Advertencia de seguridad. El archivo se carga con `weights_only=False` porque incluye hiperparámetros además de pesos. Aplica la misma precaución que con el pickle.

## Verificación de integridad

```
sha256  467ccab5cadd5a535960176f2dbf37e8081ddf2f3bffdbcd3785e932f6205453  saves/dqn_mountaincar.pt
sha256  21c3a84ac8caba6e7441a68e38f85ebc6aa90676e001d5e4e0c454f2263df095  saves/qlearning_mountaincar.pkl
```

Se comprueba con `sha256sum saves/*`. La prueba `test_agentes_guardados_alcanzan_la_bandera` verifica además que ambos archivos cargan y alcanzan la bandera.

## Referencia

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency*, 220-229. https://doi.org/10.1145/3287560.3287596
