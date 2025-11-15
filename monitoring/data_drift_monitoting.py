import pandas as pd
import numpy as np
from datetime import datetime
#For the drift measurements we will use the evidently library
from evidently import Report
from evidently.metrics import *
from evidently.presets import *

BASELINE_PATH = "../data/final/power_tetouan_city_after_EDA.csv"      
ALTERED_PATH = "../data/modified/power_tetouan_city_altered.csv" 

def simulate_shift():

    df = pd.read_csv(BASELINE_PATH)

    # Example drift simulations:
    df_drift = df.copy()

    # 1. Desplazamiento de media
    numeric_cols = df_drift.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_drift[col] = df_drift[col] + np.random.uniform(1.0, 5.0)

    # 2. Eliminando una columna al azar
    col_to_drop = np.random.choice(df_drift.columns)
    df_drift = df_drift.drop(columns=[col_to_drop])

    df_drift.to_csv(ALTERED_PATH, index=False)
    print(f"Simulated drift data saved to {ALTERED_PATH}")

def load_baseline():
    print(f"Loading original data from {BASELINE_PATH}")
    return pd.read_csv(BASELINE_PATH)

def load_altered():
    print(f"Loading altered data from {ALTERED_PATH}")
    return pd.read_csv(ALTERED_PATH)


def generate_drift_report(df_baseline,df_altered):
    '''
    Generates data drift report using the evidently library
    '''

    report = Report([
        DataSummaryPreset(),
        DataDriftPreset()
    ], include_tests=True)

    #timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    my_eval = report.run(current_data = df_altered,
                        reference_data=df_baseline)
    
    report_path = f'./reports/drift_report_{timestamp}.html'

    my_eval.save_html(report_path)

    print(f'Drift report generated at: {report_path}')


if __name__ == "__main__":
    simulate_shift()
    baseline = load_baseline()
    current = load_altered()
    generate_drift_report(baseline, current)