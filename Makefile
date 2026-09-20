# Atajos para reproducir el taller. Requiere uv (https://docs.astral.sh/uv/).
.PHONY: ayuda instalar lint pruebas evaluar diagnostico entrenar-qlearning entrenar-dqn figuras reproducir

ayuda:
	@echo "make instalar            instala dependencias exactas (uv.lock)"
	@echo "make lint                ruff sobre todo el repositorio"
	@echo "make pruebas             pruebas rapidas de los dos agentes"
	@echo "make evaluar             evalua los agentes guardados en saves/"
	@echo "make diagnostico         cuenta banderas segun la persistencia de la exploracion"
	@echo "make entrenar-qlearning  20 000 episodios, semilla 7"
	@echo "make entrenar-dqn        2 500 episodios, semilla 7 (unos 14 min en CPU)"
	@echo "make figuras             regenera figuras (tema claro y oscuro) y GIF"
	@echo "make reproducir          todo lo anterior, en orden"

instalar:
	uv sync

lint:
	uv run ruff check .

pruebas:
	uv run pytest -q

evaluar:
	uv run mountaincar load qlearning --eval
	uv run mountaincar load dqn --eval

diagnostico:
	uv run python scripts/diagnostico_exploracion.py | tee resultados/registros/diagnostico_exploracion.txt

entrenar-qlearning:
	uv run python scripts/experimento.py qlearning --episodes 20000 --chunk 500 | tee resultados/registros/qlearning_entrenamiento.txt

entrenar-dqn:
	OMP_NUM_THREADS=1 uv run python scripts/experimento.py dqn --episodes 2500 --chunk 50 | tee resultados/registros/dqn_entrenamiento.txt

figuras:
	uv run --with matplotlib python scripts/graficar.py
	uv run --with pillow python scripts/animar.py
	PYTHONPATH=scripts uv run --with matplotlib python scripts/banner.py

reproducir: instalar lint pruebas diagnostico entrenar-qlearning entrenar-dqn figuras
