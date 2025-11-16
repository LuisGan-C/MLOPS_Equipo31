FROM python:3.11-slim

WORKDIR /app

COPY mlops_equipo31/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#COPY mlops_equipo31/api/main.py .
COPY mlops_equipo31 /app/mlops_equipo31

CMD ["uvicorn", "mlops_equipo31.api.main:app", "--host", "0.0.0.0", "--port", "8000"]