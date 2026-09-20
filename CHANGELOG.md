# Registro de cambios

## 1.1.0 (2026-09-20)

Visualización.

- Figuras rehechas con una paleta validada para daltonismo y contraste, en tema claro y oscuro; el README muestra la variante que corresponde al tema del lector.
- Figuras nuevas: mapa de política, mapa de valor, diagnóstico de exploración y eficiencia en muestras.
- GIF de un episodio voraz de cada agente (`scripts/animar.py`).

## 1.0.0 (2026-09-20)

Entrega del Taller 1, Unidad 2.

- Ejercicios 1a a 1c: discretización, selección epsilon-greedy y actualización TD de Q-Learning tabular.
- Ejercicios 2a y 2b: red Q y paso de aprendizaje de DQN con red objetivo.
- Ejercicio 3: diagnóstico del fallo de exploración y corrección con exploración persistente (`explore_repeat`).
- Protocolo de medición con semilla fija, controles voraces por tramo y evaluación final independiente.
- Resultados reportados: Q-Learning -123.0 ± 16.4 (98/100); DQN -96.7 ± 7.6 (100/100).
- Organización académica del repositorio: pruebas automáticas, Makefile, `CITATION.cff`, tarjetas de modelo, lista de reproducibilidad y bibliografía en BibTeX.

## Base

`7e294ff`, commit inicial del repositorio del curso (Emilio Muñoz Pérez): CLI, bucles de entrenamiento, persistencia y enunciado de los ejercicios.
