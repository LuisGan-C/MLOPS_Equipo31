.PHONY: run exp ui

run:
	PYTHONPATH=src python -m mlops_equipo31.train

# uso: make exp N=300 D=12
exp:
	@echo ">>> Ejecutando experimento con n_estimators=$(N)  max_depth=$(D)"
	python scripts/exp.py
	PYTHONPATH=src python -m mlops_equipo31.train

ui:
	mlflow ui --port 5000

