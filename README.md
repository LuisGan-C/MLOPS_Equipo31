# 🧠 MLOPS_Equipo31

---

## ⚙️ Requerimientos

### Dependencias principales
Instala las dependencias necesarias en tu entorno (se recomienda crear un virtualenv):

```bash
pip install dvc[s3]
pip install boto3
pip install pandas numpy scikit-learn jupyter
```

### Configuración de AWS (para usar el remote S3)
Antes de usar DVC con S3, configura tus credenciales de AWS:

```bash
aws configure --profile dvc-s3
# Ingresa:
# AWS Access Key ID [None]: <tu_access_key>
# AWS Secret Access Key [None]: <tu_secret_key>
# Default region name [None]: us-east-1
# Default output format [None]: json
```

Puedes verificar que tu perfil funcione con:
```bash
aws sts get-caller-identity --profile dvc-s3
```

---

## 🧩 Configuración de DVC

DVC ya está inicializado y configurado para usar S3 como almacenamiento remoto.  
Verifícalo con:
```bash
dvc config --list --show-origin
```

Deberías ver algo como:
```
remote.storage.url=s3://dvc-bucket-mlops/mlops-equipo31
remote.storage.region=us-east-1
remote.storage.profile=dvc-s3
core.remote=storage
```

---

## 📦 Cómo agregar y versionar datos

### 1️⃣ Agregar un nuevo dataset
Coloca el archivo dentro de la carpeta correspondiente, por ejemplo:
```
data/raw/new_dataset.csv
```

Luego ejecútalo:
```bash
dvc add data/raw/new_dataset.csv
```

Esto crea el archivo `data/raw/new_dataset.csv.dvc`.

---

### 2️⃣ Guardar los cambios en Git
```bash
git add data/raw/new_dataset.csv.dvc .gitignore
git commit -m "Add new dataset version"
```

---

### 3️⃣ Subir los datos al remoto (S3)
```bash
dvc push -v
```

Esto sube los datos grandes a S3 y mantiene en Git solo los metadatos.

---

## 💾 Cómo obtener los datos (otros usuarios)

Cuando otro colaborador clone el repo:
```bash
git clone <URL_DEL_REPO>
cd MLOPS_Equipo31
dvc pull
```

Esto descarga los datos desde S3.

---

## 🧠 Flujo básico de trabajo con DVC

1. **Clonar el repo y configurar credenciales AWS**
2. **Hacer `dvc pull`** para obtener los datos
3. **Trabajar localmente** (EDA, features, modelos)
4. **Agregar nuevos datos o modelos con `dvc add`**
5. **Hacer commit y `dvc push`** para guardar los cambios
6. **Revisar versiones o DAG con:**
   ```bash
   dvc dag
   dvc exp show
   dvc exp show --html > report.html
   open report.html
   ```

---

## ⚡ Ejemplo rápido

```bash
# Añadir dataset
dvc add data/raw/energy_efficiency_modified.csv

# Confirmar en git
git add data/raw/energy_efficiency_modified.csv.dvc
git commit -m "Add energy efficiency modified dataset"

# Subir datos al S3 remoto
dvc push
```

---

## 🧰 Herramientas útiles

| Comando | Descripción |
|----------|--------------|
| `dvc status` | Muestra qué archivos están pendientes por sincronizar |
| `dvc diff` | Compara versiones entre commits |
| `dvc metrics show` | Muestra métricas definidas en pipelines |
| `dvc exp run` | Ejecuta una nueva experimentación |
| `dvc dag` | Visualiza el flujo del pipeline |
| `dvc gc` | Limpia cachés antiguas |

---

## 🔍 

## ✅ Ejecución de pruebas automatizadas

Este proyecto utiliza pytest para validar los componentes de Machine Learning. Las pruebas están organizadas en la carpeta tests/ e incluyen:

1. Pruebas unitarias: para funciones de limpieza, preprocesamiento, entrenamiento y evaluación.

2. Pruebas de integración: para validar el pipeline de extremo a extremo.

## 🔧 Requisitos previos

Instala pytest si no lo tienes:

```bash
pip install pytest
```

## ▶️ Ejecutar todas las pruebas con un solo comando

Desde la raíz del proyecto, simplemente ejecuta:

```bash
pytest -q tests/
```
Esto buscará automáticamente todos los archivos que empiezan con test_ dentro de la carpeta tests/ y ejecutará sus funciones que empiecen con test_.

---


## 📦 Serving del modelo (FastAPI + MLflow)

## Serving (FastAPI)
- GET `/health`
- POST `/predict`
- Docs: `/docs`

### Variables
- `MLFLOW_TRACKING_URI=file:$PWD/mlruns`
- `MODEL_URI=runs:/12e23d1796e74e7184f14277e21ccf64/model`

### Correr (fish)
```fish
set -x MLFLOW_TRACKING_URI file:$PWD/mlruns
set -x MODEL_URI runs:/12e23d1796e74e7184f14277e21ccf64/model
uvicorn mlops_equipo31.api.main:app --app-dir src --host 0.0.0.0 --port 8000 --reload
```


**Ejemplo de request**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "temp": 6.414,
      "hum": 74.5,
      "wind": 0.083,
      "gen_diffuse_flows": 0.07,
      "diffuse_flows": 0.085,
      "z2_power_cons": 19375.07599,
      "z3_power_cons": 20131.08434,
      "hour": 0,
      "day_of_week": 6,
      "month": 1,
      "day": 1
    }]
  }'

# fish
set -x MLFLOW_TRACKING_URI file:$PWD/mlruns
set -x MODEL_URI runs:/<run_id>/model
uvicorn mlops_equipo31.api.main:app --app-dir src --host 0.0.0.0 --port 8000 --reload
