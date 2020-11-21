import os
import pandas as pd
from mldesigntoolkit.modules.modeling import BaggingOptunaClassifier

if __name__ == "__main__":
    bath_path = os.getcwd()
    input_table = pd.read_csv(os.path.join(bath_path, r'test_data\十万样本.csv'), header=0)
    #input_table = pd.read_csv("E:\十万样本.csv", header=0)

    input_table.pop('category_str_big')
    input_table.pop('category_str_big.1')
    input_table.pop('reg_province')
    input_table.pop('reg_city')
    input_table.pop('whether_taxpayer_a')
    input_table.pop('whether_bad_executed')
    input_table.pop('company_id')

    input_table = input_table.dropna()

    df = input_table[['label1']].rename(columns={'label1': 'label'})
    y_train = df.reset_index(drop=True)

    input_table.pop('label1')
    x_train = input_table.reset_index(drop=True)

    args = {
        'variable_dict': {
            'params': {'thread_count': -1,
                       'objective': 'Logloss',
                       "eval_metric": "AUC",
                       'verbose': False,
                       'early_stopping_rounds': 4},
            'param_spaces': {'learning_rate': ('loguniform', [0.0025, 0.005, 0.01, 0.015, 0.02, 0.025]),
                             'depth': ('int', [2, 10]),
                             'colsample_bylevel': ('uniform', [0.1, 1.0]),
                             'random_strength': ('int', [1, 20]),
                             'l2_leaf_reg': ('loguniform', [0.1, 10.0]),
                             'bootstrap_type': ('categorical', ['Bayesian', 'Bernoulli', 'MVS']),
                             'boosting_type': ('categorical', ['Ordered', 'Plain']), },
            'n_splits': 5,
            'n_trials': 20,
            'trial_early_stopping': 4,
            'algorithm_early_stopping': 4,
            'n_models': 5,
            'random_state': None,
            'classifier': 'CBDTClassifier'
        }
    }
    bag = BaggingOptunaClassifier(**args)

    bag.fit(x_train, y_train)

    print(bag.predict(x_train))
