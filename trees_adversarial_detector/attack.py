import concurrent.futures
import concurrent.futures
import concurrent.futures
import json
import random
import time

import numpy as np
import sklearn
from numpy import linalg as LA
from tqdm import tqdm

from .tree_ensemble_attack.baselines.HSJA import HSJA
from .tree_ensemble_attack.baselines.OPT_attack_lf import OPT_attack_lf
from .tree_ensemble_attack.baselines.Sign_OPT_cpu import OPT_attack_sign_SGD_cpu
from .tree_ensemble_attack.baselines.cube_attack import Cube
from .tree_ensemble_attack.baselines.models_cpu import CPUModel, XGBoostModel, \
    XGBoostTestLoader

attack_list = {
    "opt": OPT_attack_lf,
    "signopt": OPT_attack_sign_SGD_cpu,
    "hsja": HSJA,
    "cube": Cube,
}


def run_attack(xi, yi, idx, num_attack, attack, norm_order):
    best_norm = np.inf
    best_adv = None
    for i in range(num_attack):
        # Fix random seed for each attack.
        random.seed(8 + i)
        np.random.seed(8 + i)

        succ, adv = attack(xi, yi)
        if not succ:
            continue
        current_norm = LA.norm(adv - xi, norm_order)
        if current_norm < best_norm:
            best_norm = current_norm
            best_adv = adv

    succ = best_adv is not None
    return succ, best_adv


def generate_hopskipjump_attack(config_path, default_num_points=100):
    import xgboost
    from art.attacks.evasion import HopSkipJump
    from art.estimators.classification import XGBoostClassifier

    with open(config_path) as json_file:
        config = json.load(json_file)

    X, y = sklearn.datasets.load_svmlight_file(config['inputs'], zero_based=True, n_features=config['num_features'])
    X = X.toarray()
    X, y = sklearn.utils.shuffle(X, y)

    total_Linf = 0.0
    total_L1 = 0.0
    total_L2 = 0.0
    total_success = 0
    num_attack = int(config['num_attack_per_point'])
    norm_order = config['norm_type']
    # -1 was used as Inf for other benchmarks
    if norm_order == -1:
        norm_order = np.inf

    xgb = xgboost.XGBClassifier()
    xgb.load_model(config['model'].replace('trees_model.json', 'trees_model_sklearn_version.json'))
    art_xgb = XGBoostClassifier(xgb, nb_features=config['num_features'], nb_classes=config['num_classes'])
    hopskipjump_attack_obj = HopSkipJump(art_xgb, targeted=False, verbose=False)

    adv_samples = {}
    attack_jobs = []
    for i in range(X.shape[0]):
        adv_samples[i] = {'idx': i, 'original_xi': X[i, :].reshape(1, -1), 'original_yi': y[i]}

        if (xgb.predict(X[i, :].reshape(1, -1)) != y[i]):
            adv_samples[i]['model_classify_correct'] = False
            continue

        adv_samples[i]['model_classify_correct'] = True
        attack_jobs.append(
            {'xi': X[i, :].reshape(1, -1), 'yi': y[i], 'i': i, 'num_attack': num_attack, 'norm_order': norm_order,
             'attack': hopskipjump_attack_obj})

    attacks_res = run_art_parallel_attacks(attack_jobs, xgb, adv_samples, config, default_num_points)

    for res in attacks_res:
        i, succ, adv = res

        adv_samples[i]['adv_succ'] = succ
        adv_samples[i]['adv_vector'] = adv
        if succ:
            if (xgb.predict(adv).reshape(1, -1) == y[i]):
                adv_samples[i]['adv_succ'] = False
                continue

            adv_samples[i]['adv_check_passed'] = True

            adv_samples[i]['L_inf_norm'] = LA.norm(adv - adv_samples[i]['original_xi'], np.inf)
            adv_samples[i]['L_1_norm'] = LA.norm(adv - adv_samples[i]['original_xi'], 1)
            adv_samples[i]['L_2_norm'] = LA.norm(adv - adv_samples[i]['original_xi'], 2)

            total_Linf += LA.norm(adv - adv_samples[i]['original_xi'], np.inf)
            total_L1 += LA.norm(adv - adv_samples[i]['original_xi'], 1)
            total_L2 += LA.norm(adv - adv_samples[i]['original_xi'], 2)
            total_success += 1

    print('Num Attacked: %d Actual Examples Tested:%d' % (X.shape[0], total_success))

    # Hack to avoid div-by-0.
    if total_success == 0:
        total_success = 1

    final_statistics = {'Norm(-1)': total_Linf / total_success,
                        'Norm(1)': total_L1 / total_success,
                        'Norm(2)': total_L2 / total_success}

    return adv_samples, final_statistics


def generate_sigopt_attack(config_path, default_num_points=100):
    import xgboost
    from art.attacks.evasion import SignOPTAttack
    from art.estimators.classification import XGBoostClassifier

    with open(config_path) as json_file:
        config = json.load(json_file)

    X, y = sklearn.datasets.load_svmlight_file(config['inputs'], zero_based=True, n_features=config['num_features'])
    X = X.toarray()
    X, y = sklearn.utils.shuffle(X, y)

    total_Linf = 0.0
    total_L1 = 0.0
    total_L2 = 0.0
    total_success = 0
    num_attack = int(config['num_attack_per_point'])
    norm_order = config['norm_type']
    # -1 was used as Inf for other benchmarks
    if norm_order == -1:
        norm_order = np.inf

    xgb = xgboost.XGBClassifier()
    xgb.load_model(config['model'].replace('trees_model.json', 'trees_model_sklearn_version.json'))
    art_xgb = XGBoostClassifier(xgb, nb_features=config['num_features'], nb_classes=config['num_classes'])
    signopt_attack_obj = SignOPTAttack(art_xgb, targeted=False, verbose=False)
    signopt_attack_obj.clip_min = None
    signopt_attack_obj.clip_max = None

    adv_samples = {}
    attack_jobs = []
    for i in range(X.shape[0]):
        adv_samples[i] = {'idx': i, 'original_xi': X[i, :].reshape(1, -1), 'original_yi': y[i]}

        if (xgb.predict(X[i, :].reshape(1, -1)) != y[i]):
            adv_samples[i]['model_classify_correct'] = False
            continue

        adv_samples[i]['model_classify_correct'] = True
        attack_jobs.append(
            {'xi': X[i, :].reshape(1, -1), 'yi': y[i], 'i': i, 'num_attack': num_attack, 'norm_order': norm_order,
             'attack': signopt_attack_obj})

    attacks_res = run_art_parallel_attacks(attack_jobs, xgb, adv_samples, config, default_num_points)

    for res in attacks_res:
        i, succ, adv = res

        adv_samples[i]['adv_succ'] = succ
        adv_samples[i]['adv_vector'] = adv
        if succ:
            if (xgb.predict(adv).reshape(1, -1) == y[i]):
                adv_samples[i]['adv_succ'] = False
                continue

            adv_samples[i]['adv_check_passed'] = True

            adv_samples[i]['L_inf_norm'] = LA.norm(adv - adv_samples[i]['original_xi'], np.inf)
            adv_samples[i]['L_1_norm'] = LA.norm(adv - adv_samples[i]['original_xi'], 1)
            adv_samples[i]['L_2_norm'] = LA.norm(adv - adv_samples[i]['original_xi'], 2)

            total_Linf += LA.norm(adv - adv_samples[i]['original_xi'], np.inf)
            total_L1 += LA.norm(adv - adv_samples[i]['original_xi'], 1)
            total_L2 += LA.norm(adv - adv_samples[i]['original_xi'], 2)
            total_success += 1

    print('Num Attacked: %d Actual Examples Tested:%d' % (X.shape[0], total_success))

    # Hack to avoid div-by-0.
    if total_success == 0:
        total_success = 1

    final_statistics = {'Norm(-1)': total_Linf / total_success,
                        'Norm(1)': total_L1 / total_success,
                        'Norm(2)': total_L2 / total_success}

    return adv_samples, final_statistics


def art_parallel_attack(params):
    adv_sample = params['attack'].generate(params['xi'].reshape(1, -1))
    return params['i'], adv_sample


# def run_art_parallel_attacks(attack_jobs, model, adv_samples, config, default_num_points):
#     count_succ = 0
#     attacks_res = []
#     with tqdm(total=len(attack_jobs)) as pbar:
#         with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
#             batch_size = 20
#             i = 0
#             while count_succ < default_num_points:
#                 future_tasks = []
#                 for attack_job in attack_jobs[i*batch_size:(i+1)*batch_size]:
#                     future_tasks.append(executor.submit(art_parallel_attack, params=attack_job))
#                 for future in concurrent.futures.as_completed(future_tasks):
#                     attack_res = future.result()
#                     i, adv_sample = attack_res
#                     if (model.predict(adv_sample).reshape(1, -1) != adv_samples[i]['original_yi']):
#                         adv_check = True
#                     else:
#                         adv_check = False
#
#                     if adv_check:
#                         count_succ += 1
#                         attacks_res.append((i, True, adv_sample))
#                     else:
#                         attacks_res.append((i, False, adv_sample))
#
#                     pbar.update(1)
#                 i+1
#
#     return attacks_res

#
# def run_art_parallel_attacks(attack_jobs, model, adv_samples, config, default_num_points):
#     count_succ = 0
#     attacks_res = []
#     with tqdm(total=len(attack_jobs)) as pbar:
#         with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
#             future_tasks = []
#             for attack_job in attack_jobs:
#                 future_tasks.append(executor.submit(art_parallel_attack, params=attack_job))
#             for future in concurrent.futures.as_completed(future_tasks):
#                 attack_res = future.result()
#                 i, adv_sample = attack_res
#                 if (model.predict(adv_sample).reshape(1, -1) != adv_samples[i]['original_yi']):
#                     adv_check = True
#                 else:
#                     adv_check = False
#
#                 if adv_check:
#                     count_succ += 1
#                     attacks_res.append((i, True, adv_sample))
#                 else:
#                     attacks_res.append((i, False, adv_sample))
#
#                 pbar.update(1)
#                 if count_succ >= default_num_points:
#                     print(f"Reached {default_num_points} successful attacks")
#                     return attacks_res
#     return attacks_res

def run_art_parallel_attacks(attack_jobs, model, adv_samples, config, default_num_points):
    count_succ = 0
    attacks_res = []
    with tqdm(total=len(attack_jobs)) as pbar:
        for attack_job in attack_jobs:
            attack_res = art_parallel_attack(attack_job)
            i, adv_sample = attack_res
            if (model.predict(adv_sample).reshape(1, -1) != adv_samples[i]['original_yi']):
                adv_check = True
            else:
                adv_check = False

            if adv_check:
                count_succ += 1
                attacks_res.append((i, True, adv_sample))
            else:
                attacks_res.append((i, False, adv_sample))

            pbar.update(1)
            if count_succ >= default_num_points:
                print(f"Reached {default_num_points} successful attacks")
                return attacks_res
    return attacks_res


def generate_adv_samples(config_path, default_num_points=100):
    with open(config_path) as json_file:
        config = json.load(json_file)
    if config['search_mode'] == 'signopt_art':
        return generate_sigopt_attack(config_path, default_num_points)
    elif config['search_mode'] == 'hsja_art':
        return generate_hopskipjump_attack(config_path, default_num_points)
    else:
        return _generate_adv_samples(config_path, default_num_points)


def _generate_adv_samples(config_path, default_num_points=100):
    with open(config_path) as json_file:
        config = json.load(json_file)
    print('Using config:', config)

    num_attack = int(config['num_attack_per_point'])
    # Cube Attack has built in support.
    if config['search_mode'] == 'cube':
        num_attack = 1

    test_loader = XGBoostTestLoader(config_path)
    norm_order = config['norm_type']
    # -1 was used as Inf for other benchmarks
    if norm_order == -1:
        norm_order = np.inf

    model = XGBoostModel(config_path, config['num_threads'])
    amodel = CPUModel(model)
    attack = attack_list[config['search_mode']](amodel, norm_order)

    total_Linf = 0.0
    total_L1 = 0.0
    total_L2 = 0.0
    total_success = 0

    num_examples = len(test_loader)
    timestart = time.time()

    adv_samples = {}

    attack_jobs = []

    # make this parallel
    for i, (xi, yi) in enumerate(test_loader):
        adv_samples[i] = {'idx': i, 'original_xi': xi, 'original_yi': yi}

        if config['num_classes'] == 2:
            if (amodel.predict_label(xi) != yi):
                adv_samples[i]['model_classify_correct'] = False
                continue
        else:
            if (np.argmax(amodel.predict_label(xi)) != yi):
                adv_samples[i]['model_classify_correct'] = False
                continue

        adv_samples[i]['model_classify_correct'] = True
        attack_jobs.append(
            {'xi': xi, 'yi': yi, 'i': i, 'num_attack': num_attack, 'attack': attack, 'norm_order': norm_order})

    attacks_res = run_parallel_attacks(attack_jobs, amodel, adv_samples, config, default_num_points)
    # attack_jobs.append(executor.submit(parallel_attack, xi=xi, yi=yi, i=i, num_attack=num_attack, attack=attack,
    #                                    norm_order=norm_order))

    # attack_res = []
    # with tqdm(total=4) as pbar:
    #     for future in concurrent.futures.as_completed(attack_jobs):
    #         attack_res.append(future.result())
    #         print("finished one")
    #         pbar.update(1)

    for res in attacks_res:
        i, succ, adv = res

        adv_samples[i]['adv_succ'] = succ
        adv_samples[i]['adv_vector'] = adv
        if succ:
            if config['num_classes'] == 2:
                adv_check = (amodel.predict_label(adv) != adv_samples[i]['original_yi'])
            else:
                adv_check = (np.argmax(amodel.predict_label(adv)) != adv_samples[i]['original_yi'])

            if not adv_check:
                adv_samples[i]['adv_succ'] = False
                continue

            adv_samples[i]['adv_check_passed'] = True
            # print(
            #     '\n===== Attack result for example %d/%d Norm(%d)=%lf time=%.4fs ====='
            #     %
            #     (i + 1, num_examples, config['norm_type'],
            #      LA.norm(adv - xi, norm_order), single_timeend - single_timestart))

            adv_samples[i]['L_inf_norm'] = LA.norm(adv - adv_samples[i]['original_xi'], np.inf)
            adv_samples[i]['L_1_norm'] = LA.norm(adv - adv_samples[i]['original_xi'], 1)
            adv_samples[i]['L_2_norm'] = LA.norm(adv - adv_samples[i]['original_xi'], 2)

            total_Linf += LA.norm(adv - adv_samples[i]['original_xi'], np.inf)
            total_L1 += LA.norm(adv - adv_samples[i]['original_xi'], 1)
            total_L2 += LA.norm(adv - adv_samples[i]['original_xi'], 2)
            total_success += 1

    print('Num Attacked: %d Actual Examples Tested:%d' % (len(test_loader), total_success))

    # Hack to avoid div-by-0.
    if total_success == 0:
        total_success = 1

    final_statistics = {'Norm(-1)': total_Linf / total_success,
                        'Norm(1)': total_L1 / total_success,
                        'Norm(2)': total_L2 / total_success}

    ### OLD NOT PARALLEL VERSION
    # for i, (xi, yi) in tqdm(enumerate(test_loader), total=num_examples, desc='Attacking examples'):
    #     adv_samples[i] = {'idx': i, 'original_xi': xi, 'original_yi': yi}
    #
    #     if config['num_classes'] == 2:
    #         if (amodel.predict_label(xi) != yi):
    #             adv_samples[i]['model_classify_correct'] = False
    #             continue
    #     else:
    #         if (np.argmax(amodel.predict_label(xi)) != yi):
    #             adv_samples[i]['model_classify_correct'] = False
    #             continue
    #
    #     adv_samples[i]['model_classify_correct'] = True
    #
    #     single_timestart = time.time()
    #     succ, adv = run_attack(xi, yi, i + 1, num_attack, attack, norm_order)
    #     adv_samples[i]['adv_succ'] = succ
    #     adv_samples[i]['adv_vector'] = adv
    #     single_timeend = time.time()
    #     if succ:
    #         if config['num_classes'] == 2:
    #             adv_check = (amodel.predict_label(adv) != yi)
    #         else:
    #             adv_check = (np.argmax(amodel.predict_label(adv)) != yi)
    #         assert adv_check, '!!!Attack report success but adv invalid!!!'
    #         adv_samples[i]['adv_check_passed'] = True
    #         # print(
    #         #     '\n===== Attack result for example %d/%d Norm(%d)=%lf time=%.4fs ====='
    #         #     %
    #         #     (i + 1, num_examples, config['norm_type'],
    #         #      LA.norm(adv - xi, norm_order), single_timeend - single_timestart))
    #
    #         adv_samples[i]['L_inf_norm'] = LA.norm(adv - xi, np.inf)
    #         adv_samples[i]['L_1_norm'] = LA.norm(adv - xi, 1)
    #         adv_samples[i]['L_2_norm'] = LA.norm(adv - xi, 2)
    #         adv_samples[i]['single_time'] = single_timeend - single_timestart
    #
    #         total_Linf += LA.norm(adv - xi, np.inf)
    #         total_L1 += LA.norm(adv - xi, 1)
    #         total_L2 += LA.norm(adv - xi, 2)
    #         total_success += 1
    #
    # timeend = time.time()
    # print('Num Attacked: %d Actual Examples Tested:%d' % (len(test_loader), total_success))
    #
    # # Hack to avoid div-by-0.
    # if total_success == 0:
    #     total_success = 1
    #
    # final_statistics = {'Norm(-1)': total_Linf / total_success,
    #                     'Norm(1)': total_L1 / total_success,
    #                     'Norm(2)': total_L2 / total_success,
    #                     'timepp': ((timeend - timestart) / total_success),
    #                     'xgb_timepp': (model.xgb_runtime / total_success),
    #                     'xgb_time_ratio': (model.xgb_runtime / (timeend - timestart))}

    return adv_samples, final_statistics


# def run_parallel_attacks(attack_jobs, model, adv_samples, config, default_num_points):
#     count_succ = 0
#     attacks_res = []
#     with tqdm(total=len(attack_jobs)) as pbar:
#         for attack_job in attack_jobs:
#             attack_res = parallel_attack(attack_job)
#             i, adv_sample = attack_res
#             if (model.predict(adv_sample).reshape(1, -1) != adv_samples[i]['original_yi']):
#                 adv_check = True
#             else:
#                 adv_check = False
#
#             if adv_check:
#                 count_succ += 1
#                 attacks_res.append((i, True, adv_sample))
#             else:
#                 attacks_res.append((i, False, adv_sample))
#
#             pbar.update(1)
#             if count_succ >= default_num_points:
#                 print(f"Reached {default_num_points} successful attacks")
#                 return attacks_res
#     return attacks_res


def run_parallel_attacks(attack_jobs, amodel, adv_samples, config, default_num_points):
    count_succ = 0
    attacks_res = []
    with tqdm(total=len(attack_jobs)) as pbar:
        for attack_job in attack_jobs:
            attack_res = parallel_attack(attack_job)
            i, succ, adv_sample = attack_res
            if succ:
                if config['num_classes'] == 2:
                    adv_check = (amodel.predict_label(adv_sample) != adv_samples[i]['original_yi'])
                else:
                    adv_check = (np.argmax(amodel.predict_label(adv_sample)) != adv_samples[i]['original_yi'])

                if adv_check:
                    count_succ += 1
                    attacks_res.append((i, succ, adv_sample))
                else:
                    attacks_res.append((i, False, adv_sample))

            else:
                attacks_res.append(attack_res)
            pbar.update(1)
            if count_succ >= default_num_points:
                print(f"Reached {default_num_points} successful attacks")
                return attacks_res

    return attacks_res

#
# def run_parallel_attacks(attack_jobs, amodel, adv_samples, config, default_num_points):
#     count_succ = 0
#     attacks_res = []
#     with tqdm(total=len(attack_jobs)) as pbar:
#         with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
#             future_tasks = []
#             for attack_job in attack_jobs:
#                 future_tasks.append(executor.submit(parallel_attack, params=attack_job))
#             for future in concurrent.futures.as_completed(future_tasks):
#                 attack_res = future.result()
#                 i, succ, adv_sample = attack_res
#                 if succ:
#                     if config['num_classes'] == 2:
#                         adv_check = (amodel.predict_label(adv_sample) != adv_samples[i]['original_yi'])
#                     else:
#                         adv_check = (np.argmax(amodel.predict_label(adv_sample)) != adv_samples[i]['original_yi'])
#
#                     if adv_check:
#                         count_succ += 1
#                         attacks_res.append((i, succ, adv_sample))
#                     else:
#                         attacks_res.append((i, False, adv_sample))
#
#                 else:
#                     attacks_res.append(attack_res)
#                 pbar.update(1)
#                 if count_succ >= default_num_points:
#                     print(f"Reached {default_num_points} successful attacks")
#                     return attacks_res
#
#     return attacks_res


def parallel_attack(params):
    succ, adv_sample = run_attack(params['xi'], params['yi'], params['i'], params['num_attack'], params['attack'],
                                  params['norm_order'])
    return params['i'], succ, adv_sample

#
#
# def check_simple_detector(dataset_name, directory, num_features):
#     X_normal_original_train, y_normal_original_train = sklearn.datasets.load_svmlight_file(
#         f'{directory}/{dataset_name}_train_svm_data.svmlight', n_features=num_features)
#
#     adv_results_train = joblib.load(f'{directory}/{dataset_name}_train_adv_results.jblib')
#     X_original_adv_train = []
#     for sample in adv_results_train:
#         if adv_results_train[sample]['model_classify_correct'] and adv_results_train[sample]['adv_succ']:
#             X_original_adv_train.append(adv_results_train[sample]['adv_vector'])
#     X_original_adv_stacked_train = np.vstack(X_original_adv_train)
#     X_original_normal_stacked_train = np.array(X_normal_original_train.todense())
#
#     y_original_noraml_and_adv_train = np.hstack(
#         [np.zeros(len(X_original_normal_stacked_train)), np.ones(len(X_original_adv_stacked_train))])
#     X_original_normal_and_adv_train = np.vstack([X_original_normal_stacked_train, X_original_adv_stacked_train])
#
#     X_normal_original_test, y_normal_original_test = sklearn.datasets.load_svmlight_file(
#         f'{directory}/{dataset_name}_test_svm_data.svmlight', n_features=num_features)
#
#     adv_results_test = joblib.load(f'{directory}/{dataset_name}_test_adv_results.jblib')
#     X_original_adv_test = []
#     for sample in adv_results_test:
#         if adv_results_test[sample]['model_classify_correct'] and adv_results_test[sample]['adv_succ']:
#             X_original_adv_test.append(adv_results_test[sample]['adv_vector'])
#     X_original_adv_stacked_test = np.vstack(X_original_adv_test)
#     X_original_normal_stacked_test = np.array(X_normal_original_test.todense())
#
#     y_original_noraml_and_adv_test = np.hstack(
#         [np.zeros(len(X_original_normal_stacked_test)), np.ones(len(X_original_adv_stacked_test))])
#     X_original_normal_and_adv_test = np.vstack([X_original_normal_stacked_test, X_original_adv_stacked_test])
#
#     print(f"Number of normal samples train : {len(X_original_normal_stacked_train)}")
#     print(f"Number of normal samples test : {len(X_original_normal_stacked_test)}")
#     print(f"Number of adv samples train : {len(X_original_adv_train)}")
#     print(f"Number of adv samples test : {len(X_original_adv_test)}")
#
#     # xgb_original = xgboost.XGBClassifier(n_jobs=-1, n_estimators=10, max_depth=3).fit(X_original_normal_and_adv_train,
#     #                                                                                  y_original_noraml_and_adv_train)
#     xgb_original = FaissKNeighbors(k=10, faiss=faiss)
#     xgb_original.fit(X_original_normal_and_adv_train.astype('float32'),
#                      y_original_noraml_and_adv_train.astype('float32'))
#
#     print('\n\n')
#     print(
#         f"Original Train Predicted Adv: {np.count_nonzero(xgb_original.predict(X_original_normal_and_adv_train.astype('float32')))}")
#     # print(
#     #     f"Original Train ROC AUC: {roc_auc_score(y_original_noraml_and_adv_train, xgb_original.predict_proba(X_original_normal_and_adv_train.astype('float32'))[:, 1])}")
#     print(
#         f"Original Train Recall: {recall_score(y_original_noraml_and_adv_train, xgb_original.predict(X_original_normal_and_adv_train.astype('float32')))}")
#     print(
#         f"Original Train Precision: {precision_score(y_original_noraml_and_adv_train, xgb_original.predict(X_original_normal_and_adv_train.astype('float32')))}")
#     print('\n\n')
#     print(
#         f"Original Test Predicted Adv: {np.count_nonzero(xgb_original.predict(X_original_normal_and_adv_test.astype('float32')))}")
#     # print(
#     #     f"Original Test ROC AUC: {roc_auc_score(y_original_noraml_and_adv_test, xgb_original.predict(X_original_normal_and_adv_test)[:, 1])}")
#     print(
#         f"Original Test Recall: {recall_score(y_original_noraml_and_adv_test, xgb_original.predict(X_original_normal_and_adv_test.astype('float32')))}")
#     print(
#         f"Original Test Precision: {precision_score(y_original_noraml_and_adv_test, xgb_original.predict(X_original_normal_and_adv_test.astype('float32')))}")
#     print('\n\n')
#
#     embedded_data = joblib.load(f'{directory}/{dataset_name}_embedded_vectors_with_adv.jblib')
#     is_test = joblib.load(f'{directory}/is_test.jblib')
#     normal_samples_amount = len(X_original_normal_stacked_train) + len(X_original_normal_stacked_test)
#
#     X_embedded_normal = embedded_data[:normal_samples_amount]
#     X_embedded_adv = embedded_data[normal_samples_amount:]
#
#     X_embedded_normal_stacked_train = X_embedded_normal[~is_test]
#     X_embedded_normal_stacked_test = X_embedded_normal[is_test]
#
#     X_embedded_adv_stacked_train = X_embedded_adv[:len(X_original_adv_train)]
#     X_embedded_adv_stacked_test = X_embedded_adv[len(X_original_adv_train):]
#
#     y_embedded_noraml_and_adv_train = np.hstack(
#         [np.zeros(len(X_embedded_normal_stacked_train)), np.ones(len(X_embedded_adv_stacked_train))])
#     X_embedded_normal_and_adv_train = np.vstack([X_embedded_normal_stacked_train, X_embedded_adv_stacked_train])
#
#     y_embedded_noraml_and_adv_test = np.hstack(
#         [np.zeros(len(X_embedded_normal_stacked_test)), np.ones(len(X_embedded_adv_stacked_test))])
#     X_embedded_normal_and_adv_test = np.vstack([X_embedded_normal_stacked_test, X_embedded_adv_stacked_test])
#
#     # xgb_embedded = xgboost.XGBClassifier(n_jobs=-1, n_estimators=10, max_depth=3).fit(X_embedded_normal_and_adv_train,
#     #                                                                                  y_embedded_noraml_and_adv_train)
#     xgb_embedded = FaissKNeighbors(k=10, faiss=faiss)
#     xgb_embedded.fit(X_embedded_normal_and_adv_train.astype('float32'),
#                      y_embedded_noraml_and_adv_train.astype('float32'))
#
#     print('\n\n')
#     print(
#         f"Embedded Train Predicted Adv: {np.count_nonzero(xgb_embedded.predict(X_embedded_normal_and_adv_train.astype('float32')))}")
#     # print(
#     #     f"Embedded Train ROC AUC: {roc_auc_score(y_embedded_noraml_and_adv_train, xgb_embedded.predict_proba(X_embedded_normal_and_adv_train.astype('float32'))[:, 1])}")
#     print(
#         f"Embedded Train Recall: {recall_score(y_embedded_noraml_and_adv_train, xgb_embedded.predict(X_embedded_normal_and_adv_train.astype('float32')))}")
#     print(
#         f"Embedded Train Precision: {precision_score(y_embedded_noraml_and_adv_train, xgb_embedded.predict(X_embedded_normal_and_adv_train.astype('float32')))}")
#     print('\n\n')
#     print(
#         f"Embedded Test Predicted Adv: {np.count_nonzero(xgb_embedded.predict(X_embedded_normal_and_adv_test.astype('float32')))}")
#     # print(
#     #     f"Embedded Test ROC AUC: {roc_auc_score(y_embedded_noraml_and_adv_test, xgb_embedded.predict_proba(X_embedded_normal_and_adv_test)[:, 1])}")
#     print(
#         f"Embedded Test Recall: {recall_score(y_embedded_noraml_and_adv_test, xgb_embedded.predict(X_embedded_normal_and_adv_test.astype('float32')))}")
#     print(
#         f"Embedded Test Precision: {precision_score(y_embedded_noraml_and_adv_test, xgb_embedded.predict(X_embedded_normal_and_adv_test.astype('float32')))}")
#     print('\n\n')
