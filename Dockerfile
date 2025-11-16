#Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.ccds.txt .
RUN pip install --no-cache-dir -r requirements.ccds.txt
#COPY data/ ./data/
#COPY mlruns /mlruns
COPY .dvc/ ./.dvc/
COPY src /app/src
#CMD ["python", "src/mlops_equipo31/scripts/train_and_log.py", \
#     "--csv", "data/final/power_tetouan_city_after_EDA.csv", \
#     "--experiment", "equipo31-remote"]
#CMD ["dvc", "repro"]
# End of Dockerfile
