import pandas as pd

def generate_sql_dump(df, table_name):
    columns = ',\n'.join([f'"{col}" {dtype}' for col, dtype in zip(
        df.columns,
        ['TEXT' if df[col].dtype == 'object' else 'DOUBLE PRECISION' for col in df.columns]
    )])
    create_table = f'CREATE TABLE {table_name} (\n{columns}\n);\n'

    insert_statements = []
    for _, row in df.iterrows():
        values = ', '.join([f"'{str(value).replace('\'', '\'\'')}'" if isinstance(value, str) else str(value)
                            for value in row])
        insert_statements.append(f"INSERT INTO {table_name} VALUES ({values});")

    return create_table + '\n'.join(insert_statements) + '\n'

datasets = {
    "Atlanta": {
        "dataset": 'Atlanta_Dataset.csv',
        "predictions": 'Atlanta_predictions.csv'
    },
    "Denver": {
        "dataset": 'Denver_Dataset.csv',
        "predictions": 'Denver_predictions.csv'
    },
    "LosAngeles": {
        "dataset": 'LosAngeles_Dataset.csv',
        "predictions": 'LosAngeles_predictions.csv'
    },
    "Phoenix": {
        "dataset": 'Phoenix_Dataset.csv',
        "predictions": 'Phoenix_predictions.csv'
    },
    "Pittsburgh": {
        "dataset": 'Pittsburgh_Dataset.csv',
        "predictions": 'Pittsburgh_predictions.csv'
    },
    "Seattle": {
        "dataset": 'Seattle_Dataset.csv',
        "predictions": 'Seattle_predictions.csv'
    }
}

all_datasets = []
all_predictions = []

for city, paths in datasets.items():
    dataset = pd.read_csv(paths["dataset"])
    predictions = pd.read_csv(paths["predictions"])
    dataset['City'] = city
    predictions['City'] = city
    all_datasets.append(dataset)
    all_predictions.append(predictions)

combined_datasets = pd.concat(all_datasets, ignore_index=True)
combined_predictions = pd.concat(all_predictions, ignore_index=True)

datasets_sql = generate_sql_dump(combined_datasets, 'datasets')
predictions_sql = generate_sql_dump(combined_predictions, 'predictions')

with open('db_dump.sql', 'w') as f:
    f.write(datasets_sql)
    f.write('\n')
    f.write(predictions_sql)
