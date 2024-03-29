############################################################
### Forked from https://github.com/cmhcbb/attackbox
############################################################

from Sign_OPT_cpu import OPT_attack_sign_SGD_cpu
from HSJA import HSJA
from OPT_attack_lf import OPT_attack_lf
from cube_attack import Cube
from models_cpu import CPUModel, XGBoostModel, XGBoostTestLoader
import os, argparse
import time
import random
from numpy import linalg as LA
import numpy as np
import json
import joblib
import xgboost
from sklearn.tree import DecisionTreeClassifier


def check_simple_detector(dataset_name):
    X_normal_original_train, y_normal_original_train = load_svmlight_file(f'{dataset_name}_train_svm_data.svmlight')
    
    adv_results_train = joblib.load(f'{dataset_name}_train_adv_results.jblib')
    X_original_adv_train = []
    for sample in adv_results_train:
        if adv_results_train[sample]['model_classify_correct'] and adv_results_train[sample]['adv_succ']:
            X_original_adv_train.append(adv_results_train[sample]['adv_vector'])
    X_original_adv_stacked_train = np.vstack(X_original_adv_train)
    X_original_normal_stacked_train = np.array(X_normal_original_train.todense())
    
    y_original_noraml_and_adv_train  = np.hstack([np.zeros(len(X_original_normal_stacked_train)), np.ones(len(X_original_adv_stacked_train))])
    X_original_normal_and_adv_train = np.vstack([X_original_normal_stacked_train, X_original_adv_stacked_train])
    
    X_normal_original_test, y_normal_original_test = load_svmlight_file(f'{dataset_name}_test_svm_data.svmlight')

    adv_results_test = joblib.load(f'{dataset_name}_test_adv_results.jblib')
    X_original_adv_test = []
    for sample in adv_results_test:
        if adv_results_test[sample]['model_classify_correct'] and adv_results_test[sample]['adv_succ']:
            X_original_adv_test.append(adv_results_test[sample]['adv_vector'])
    X_original_adv_stacked_test = np.vstack(X_original_adv_test)
    X_original_normal_stacked_test = np.array(X_normal_original_test.todense())
    
    y_original_noraml_and_adv_test  = np.hstack([np.zeros(len(X_original_normal_stacked_test)), np.ones(len(X_original_adv_stacked_test))])
    X_original_normal_and_adv_test = np.vstack([X_original_normal_stacked_test, X_original_adv_stacked_test])
    
    print(f"Number of normal samples train : {len(X_original_normal_stacked_train)}")
    print(f"Number of normal samples test : {len(X_original_normal_stacked_test)}")
    print(f"Number of adv samples train : {len(X_original_adv_train)}")
    print(f"Number of adv samples test : {len(X_original_adv_test)}")

    xgb_original = xgboost.XGBClassifier(n_jobs=-1, n_estimators=5, max_depth=2).fit(X_original_normal_and_adv_train, y_original_noraml_and_adv_train)
    
    print('\n\n')
    print(f"Original Train Predicted Adv: {np.count_nonzero(xgb_original.predict(X_original_normal_and_adv_train))}")
    print(f"Original Train ROC AUC: {roc_auc_score(y_original_noraml_and_adv_train, xgb_original.predict_proba(X_original_normal_and_adv_train)[:, 1])}")
    print(f"Original Train Recall: {recall_score(y_original_noraml_and_adv_train, xgb_original.predict(X_original_normal_and_adv_train))}")
    print(f"Original Train Precision: {precision_score(y_original_noraml_and_adv_train, xgb_original.predict(X_original_normal_and_adv_train))}")
    print('\n\n')
    print(f"Original Test Predicted Adv: {np.count_nonzero(xgb_original.predict(X_original_normal_and_adv_test))}")
    print(f"Original Test ROC AUC: {roc_auc_score(y_original_noraml_and_adv_test, xgb_original.predict_proba(X_original_normal_and_adv_test)[:, 1])}")
    print(f"Original Test Recall: {recall_score(y_original_noraml_and_adv_test, xgb_original.predict(X_original_normal_and_adv_test))}")
    print(f"Original Test Precision: {precision_score(y_original_noraml_and_adv_test, xgb_original.predict(X_original_normal_and_adv_test))}")
    print('\n\n')

    embedded_data = joblib.load(f'{dataset_name}_embedded_vectors_with_adv.jblib')
    is_test = joblib.load('is_test.jblib')
    normal_samples_amount = len(X_original_normal_stacked_train) + len(X_original_normal_stacked_test)

    X_embedded_normal = embedded_data[:normal_samples_amount]
    X_embedded_adv = embedded_data[normal_samples_amount:]
    
    X_embedded_normal_stacked_train = X_embedded_normal[~is_test]
    X_embedded_normal_stacked_test = X_embedded_normal[is_test]
    
    X_embedded_adv_stacked_train = X_embedded_adv[:len(X_original_adv_train)]
    X_embedded_adv_stacked_test = X_embedded_adv[len(X_original_adv_train):]
    
    
    y_embedded_noraml_and_adv_train  = np.hstack([np.zeros(len(X_embedded_normal_stacked_train)), np.ones(len(X_embedded_adv_stacked_train))])
    X_embedded_normal_and_adv_train = np.vstack([X_embedded_normal_stacked_train, X_embedded_adv_stacked_train])
        
    y_embedded_noraml_and_adv_test  = np.hstack([np.zeros(len(X_embedded_normal_stacked_test)), np.ones(len(X_embedded_adv_stacked_test))])
    X_embedded_normal_and_adv_test = np.vstack([X_embedded_normal_stacked_test, X_embedded_adv_stacked_test])
    
    xgb_embedded = xgboost.XGBClassifier(n_jobs=-1, n_estimators=5, max_depth=2).fit(X_embedded_normal_and_adv_train, y_embedded_noraml_and_adv_train)

    print('\n\n')
    print(f"Embedded Train Predicted Adv: {np.count_nonzero(xgb_embedded.predict(X_embedded_normal_and_adv_train))}")
    print(f"Embedded Train ROC AUC: {roc_auc_score(y_embedded_noraml_and_adv_train, xgb_embedded.predict_proba(X_embedded_normal_and_adv_train)[:, 1])}")
    print(f"Embedded Train Recall: {recall_score(y_embedded_noraml_and_adv_train, xgb_embedded.predict(X_embedded_normal_and_adv_train))}")
    print(f"Embedded Train Precision: {precision_score(y_embedded_noraml_and_adv_train, xgb_embedded.predict(X_embedded_normal_and_adv_train))}")
    print('\n\n')
    print(f"Embedded Test Predicted Adv: {np.count_nonzero(xgb_embedded.predict(X_embedded_normal_and_adv_test))}")
    print(f"Embedded Test ROC AUC: {roc_auc_score(y_embedded_noraml_and_adv_test, xgb_embedded.predict_proba(X_embedded_normal_and_adv_test)[:, 1])}")
    print(f"Embedded Test Recall: {recall_score(y_embedded_noraml_and_adv_test, xgb_embedded.predict(X_embedded_normal_and_adv_test))}")
    print(f"Embedded Test Precision: {precision_score(y_embedded_noraml_and_adv_test, xgb_embedded.predict(X_embedded_normal_and_adv_test))}")
    print('\n\n')


config_path = r"C:\Msc\Thesis\lt_attack\tree-ensemble-attack-main\my_attack\voice_test_attack_config.json"

attack_list = {
    "opt": OPT_attack_lf,
    "signopt": OPT_attack_sign_SGD_cpu,
    "hsja": HSJA,
    "cube": Cube,
}

with open(config_path) as json_file:
    config = json.load(json_file)

print('Using config:', config)

num_attack = int(config['num_attack_per_point'])
# Cube Attack has built in support.
if config['search_mode'] == 'cube':
    num_attack = 1

print('Using num_attack:', num_attack)

test_loader = XGBoostTestLoader(config_path)
norm_order = config['norm_type']
# -1 was used as Inf for other benchmarks
if norm_order == -1:
    norm_order = np.inf

model = XGBoostModel(config_path, config['num_threads'])
amodel = CPUModel(model)
attack = attack_list[config['search_mode']](amodel, norm_order)


def run_attack(xi, yi, idx):
    best_norm = np.inf
    best_adv = None
    for i in range(num_attack):
        # Fix random seed for each attack.
        random.seed(8 + i)
        np.random.seed(8 + i)

        succ, adv = attack(xi, yi)
        if not succ:
            print('!!!Failed on example %d attack %d' % (idx, i + 1))
            continue
        current_norm = LA.norm(adv - xi, norm_order)
        print('Example %d attack %d: Norm=%.4f' % (idx, i + 1, current_norm))
        if current_norm < best_norm:
            best_norm = current_norm
            best_adv = adv

    succ = best_adv is not None
    return succ, best_adv


total_Linf = 0.0
total_L1 = 0.0
total_L2 = 0.0
total_success = 0

num_examples = len(test_loader)
print('Attacking %d examples...' % num_examples)
timestart = time.time()

adv_samples = {}

for i, (xi, yi) in enumerate(test_loader):
    print(f"sample batch: {i}")
    adv_samples[i] = {'idx': i, 'original_xi': xi, 'original_yi': yi}

    if (amodel.predict_label(xi) != yi):
        print(f"Fail to classify example {i+1}. No need to attack.")
        adv_samples[i]['model_classify_correct'] = False 
        continue
    adv_samples[i]['model_classify_correct'] = True

    #adv=attack(xi,yi, 0.2)
    single_timestart = time.time()
    succ, adv = run_attack(xi, yi, i + 1)
    adv_samples[i]['adv_succ'] = succ
    adv_samples[i]['adv_vector'] = adv
    single_timeend = time.time()
    if succ:
        adv_check = (amodel.predict_label(adv) != yi)
        assert adv_check, '!!!Attack report success but adv invalid!!!'
        adv_samples[i]['adv_check_passed'] = True
        print(
            '\n===== Attack result for example %d/%d Norm(%d)=%lf time=%.4fs ====='
            %
            (i + 1, num_examples, config['norm_type'],
             LA.norm(adv - xi, norm_order), single_timeend - single_timestart))
        
        adv_samples[i]['L_inf_norm'] = LA.norm(adv - xi, np.inf)
        adv_samples[i]['L_1_norm'] = LA.norm(adv - xi, 1)
        adv_samples[i]['L_2_norm'] = LA.norm(adv - xi, 2)
             
        total_Linf += LA.norm(adv - xi, np.inf)
        total_L1 += LA.norm(adv - xi, 1)
        total_L2 += LA.norm(adv - xi, 2)
        total_success += 1
    else:
        print('!!!Failed on example %d' % (i + 1))

timeend = time.time()

print('*******************************')
print('*******************************')
print('Results for config:', config_path)
print('Num Attacked: %d Actual Examples Tested:%d' %
      (len(test_loader), total_success))

# Hack to avoid div-by-0.
if total_success == 0:
    total_success = 1

print('Norm(-1)=%.4f' % (total_Linf / total_success))
print('Norm(1)=%.4f' % (total_L1 / total_success))
print('Norm(2)=%.4f' % (total_L2 / total_success))
print('Time per point: %.4f s' % ((timeend - timestart) / total_success))
print('XGB Time per point: %.4f s' % (model.xgb_runtime / total_success))
print('XGB Time ratio: %.4f' % (model.xgb_runtime / (timeend - timestart)))

final_statistics = {'Norm(-1)': total_Linf / total_success, 'Norm(1)': total_L1 / total_success, 'Norm(2)': total_L2 / total_success}

joblib.dump(adv_samples, r"C:\Msc\Thesis\lt_attack\tree-ensemble-attack-main\my_attack\voice_test_adv_results.jblib")
joblib.dump(final_statistics, r"C:\Msc\Thesis\lt_attack\tree-ensemble-attack-main\my_attack\voice_test_final_statistics.jblib")