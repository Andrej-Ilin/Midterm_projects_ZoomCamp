import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression, Ridge
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, auc, recall_score, precision_score, f1_score, multilabel_confusion_matrix
from sklearn import metrics
import pickle
import warnings
warnings.filterwarnings("ignore")

# Data train load
data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

bin_cols = ['dual_sim', 'blue', 'four_g', 'three_g', 'touch_screen', 'wifi', ]
num_cols = ['battery_power', 'mobile_wt', 'int_memory', 'px_height', 'px_width', 'ram']
cat_cols = ['clock_speed', 'fc', 'm_dep', 'n_cores', 'pc', 'sc_h', 'sc_w', 'talk_time']
target = ['price_range']

df_train, df_val, y_train, y_val = train_test_split(data[cat_cols + num_cols + bin_cols],
                                                    data['price_range'], test_size=0.2, random_state=1)
# Prepare

def prepare_data(df_train, df_val, test):
    dv = DictVectorizer(sparse=False)

    train_dict = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)

    val_dict = df_val.to_dict(orient='records')
    X_val = dv.transform(val_dict)

    test_dict = test.to_dict(orient='records')
    X_test = dv.transform(test_dict)
    return dv, X_train, X_val, X_test

dv, X_train, X_val, X_test = prepare_data(df_train, df_val, test)

# training

tunned_model = CatBoostClassifier(
    random_seed=42,
    iterations=1000,
    learning_rate=0.03,
    l2_leaf_reg=3,
    bagging_temperature=1,
    random_strength=1,
    one_hot_max_size=2,
    leaf_estimation_method='Newton'
)

tunned_model.fit(
    X_train, y_train,
    verbose=False,
    eval_set=(X_val, y_val),
    plot=True

)
y_pred = tunned_model.predict(X_val)
print(f'classification_report, {metrics.classification_report(y_pred, y_val)}')

# training final model
print('training the final model')
X = np.vstack([X_train, X_val])
y = np.hstack([y_train, y_val])

best_model = CatBoostClassifier(
    random_seed=42,
    iterations=int(tunned_model.tree_count_ * 1.2)
)

best_model.fit(
    X, y,
    verbose=100
)

output_file = f'best_model.bin'
# safe model
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, best_model), f_out)

print(f'the model is saved to {output_file}')
