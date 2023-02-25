from pycaret.classification import *
import pandas as pd
import os

path_home = os.getcwd()

path_data = path_home + '/datasets/datasets/salary/salary.csv'


def get_data(path_data:str):
    df = pd.read_csv(path_data)
    return df


if __name__ == '__main__':
    variables = ['age', 'fnlwgt', 'education-num', 
                 'capital-gain', 'capital-loss', 
                 'hours-per-week']
    target = 'salary'
    # Datos + limpieza
    df = get_data(path_data)
    df.columns = [k.strip() for k in df.columns]
    df[target] = df[target].apply(lambda k: str(k).strip()).map({'<=50K':0, '>50K':1})
    df = df[variables+[target]].copy()

    # split
    data = df.sample(frac=0.9, random_state=123)
    data2evaluate = df.drop(data.index)

    data = data.reset_index(drop=True)
    data2evaluate = data2evaluate.reset_index(drop=True)

    data2evaluate.to_csv('evaluacion.csv',index=False)

    # Pre-procesamiento
    dfp = setup(data = data, target=target, session_id = 123, experiment_name = 'salary_1', silent = True)

    # Modelos
    best_model = compare_models()

    # Modelo tuneado
    tuned_best = tune_model(best_model)

    # GUARDAR MODELO
    final_best = finalize_model(tuned_best)
    save_model(final_best,'salary')

    # GENERAR API FASTAPI
    create_api(final_best, 'salary_docker')

    # CREAR DOCKER
    create_docker('salary_docker')
