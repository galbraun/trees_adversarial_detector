import numpy as np
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRFClassifier


class ModelUnkownException(Exception):
    pass


def train_tree_model(model_type, X, y):
    if model_type == 'XGBoost':
        model_params = {'random_state': 84, 'n_jobs': -1, 'n_estimators': 20, 'max_depth': 6}
        return train_xgboost_model(X, y, model_params)

    if model_type == 'RandomForest':
        #model_params = {'random_state': 84, 'n_jobs': -1, 'colsample_bytree': 0.75, 'max_depth': 8, 'n_estimators': 25}
        model_params = {'random_state': 84, 'n_jobs': -1, 'colsample_by_tree': 0.75}
        return train_randomforest_model(X, y, model_params)

    raise ModelUnkownException(f'here is no familiar model with the name {model_type}')


def train_xgboost_model(X, y, model_params):
    xgb = XGBClassifier(**model_params)
    xgb.fit(X, y)
    return xgb


def train_randomforest_model(X, y, RF_params):
    rf = XGBRFClassifier(**RF_params)
    rf.fit(X, y)
    return rf


def extract_nodes_splits(model):
    df = model.get_booster().trees_to_dataframe()
    df_splits = df.loc[df.Feature != 'Leaf']
    df_splits['feature_index'] = df_splits.Feature.str.replace("f", "")
    return df_splits




def _root_to_leaf_route(df, stack, routes):
    current_node = df.loc[df.Node == stack[-1]]
    if current_node.Feature.values[0] == 'Leaf':
        routes[current_node.Node.values[0]] = list(stack)
        stack.pop()
        return

    stack.append(int(current_node.Yes.values[0].split('-')[1]))
    _root_to_leaf_route(df, stack, routes)

    stack.append(int(current_node.No.values[0].split('-')[1]))
    _root_to_leaf_route(df, stack, routes)

    stack.pop()
    return


def extract_root_to_leaf_routes_for_forest(xgb):
    routes_forest = {}
    # Changed this for multiclass option.. need to further investigate! TODO: keep investigate
    #for i in range(xgb.n_estimators):
    for i in range(len(xgb.get_booster().get_dump())):
        routes_forest[i] = extract_root_to_leaf_routes_for_tree(xgb, i)
    return routes_forest


def extract_root_to_leaf_routes_for_tree(xgb, tree_index):
    df = xgb.get_booster().trees_to_dataframe()
    df = df.loc[df.Tree == tree_index].set_index('ID')

    routes = {}
    stack = [0]
    _root_to_leaf_route(df, stack, routes)
    return routes


def extract_nodes_samples(X, model_obj):
    forest_routes = extract_root_to_leaf_routes_for_forest(model_obj)
    node_to_samples = {node: [] for node in model_obj.get_booster().trees_to_dataframe().ID}

    for i in tqdm(range(X.shape[0]), total=X.shape[0], desc="Calculating samples distribution in trees"):
        nodes = model_obj.apply(X[i].reshape(1, -1))[0]
        # if there is only one estimator
        if type(nodes) == np.int32:
            nodes = [nodes]

        # Changed this for multiclass option.. need to further investigate! TODO: keep investigate
        #for l in range(model_obj.n_estimators):
        for l in range(len(model_obj.get_booster().get_dump())):
            for n in forest_routes[l][nodes[l]]:
                node_to_samples[f'{l}-{n}'].append(i)


    return node_to_samples
