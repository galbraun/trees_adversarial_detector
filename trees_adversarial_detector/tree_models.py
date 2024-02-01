import numpy as np
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRFClassifier


class ModelUnkownException(Exception):
    pass


def train_tree_model(model_type, X, y, model_params=None):
    if model_type == 'XGBoost':
        if model_params is None:
            model_params = {'random_state': 84, 'n_jobs': -1, 'n_estimators': 10, 'max_depth': 7, 'verbosity': 3}
            # model_params = {'random_state': 84, 'n_jobs': -1}
        return train_xgboost_model(X, y, model_params)

    if model_type == 'RandomForest':
        if model_params is None:
            # model_params = {'random_state': 84, 'n_jobs': -1, 'colsample_bytree': 0.75, 'max_depth': 8, 'n_estimators': 25}
            model_params = {'random_state': 84, 'n_jobs': -1, 'colsample_bytree': 0.5, 'max_depth': 10, 'verbosity': 3,
                            'n_estimators': 30}
            # model_params = {'random_state': 84, 'n_jobs': -1, 'colsample_by_tree': 0.75}
        return train_randomforest_model(X, y, model_params)

    raise ModelUnkownException(f'there is no familiar model with the name {model_type}')


def train_xgboost_model(X, y, model_params):
    xgb = XGBClassifier(**model_params)
    xgb.fit(X, y)
    return xgb


def train_randomforest_model(X, y, RF_params):
    rf = XGBRFClassifier(**RF_params)
    rf.fit(X, y)
    return rf


def extract_nodes_splits(model):
    """
        Given a XGBoost model, extract all of the possible splits in the model
    :param model:
    :return:
    """
    # Extract the model information to a dataframe
    df = model.get_booster().trees_to_dataframe()

    # Filter only the splits from the nodes
    df_splits = df.loc[df.Feature != 'Leaf']

    # TODO: Remove the f from the feature name - this is relevant only when no column names passed so for now commenting this out
    df_splits['feature_index'] = [int(v) for v in df_splits.Feature.str.replace("f", "")]
    return df_splits


def extract_nodes_splits_per_tree(model):
    # Extract the model information to a dataframe
    df = model.get_booster().trees_to_dataframe()

    # Filter only the splits from the nodes
    df_splits = df.loc[df.Feature != 'Leaf']

    # TODO: Remove the f from the feature name - this is relevant only when no column names passed so for now commenting this out
    df_splits['feature_index'] = [int(v) for v in df_splits.Feature.str.replace("f", "")]

    splitted_by_tree = dict(tuple(df_splits.groupby('Tree')))
    return splitted_by_tree


def extract_root_to_leaf_routes_for_forest(xgb):
    """
        Given a forest of trees - extract all of the possible roots in all of the trees in the forest
    :param xgb:
    :return:
    """
    routes_forest = {}
    # Changed this for multiclass option.. need to further investigate! TODO: keep investigate
    # for i in range(xgb.n_estimators):
    for i in range(len(xgb.get_booster().get_dump())):
        routes_forest[i] = extract_root_to_leaf_routes_for_tree(xgb, i)
    return routes_forest


def extract_root_to_leaf_routes_for_tree(xgb, tree_index):
    """
        Given an XGBoost model and a tree index - extract all the root to leaf possible routes in the tree
    :param xgb:
    :param tree_index:
    :return:
    """
    # Extract from entire enemble the relevant tree nodes and set the ID as the index
    df = xgb.get_booster().trees_to_dataframe()
    df = df.loc[df.Tree == tree_index].set_index('ID')

    # Collect the traversed nodes
    routes = {}
    stack = [0]
    _root_to_leaf_route(df, stack, routes)
    return routes


def _root_to_leaf_route(df, stack, routes):
    """
        Recursive function to extract all of the possible routes in a tree

    :param df:
    :param stack:
    :param routes:
    :return:
    """
    # Get the current node in the df
    current_node = df.loc[df.Node == stack[-1]]

    # If we reached a leaf add the route to the routes collection and pop pop one level up
    if current_node.Feature.values[0] == 'Leaf':
        routes[current_node.Node.values[0]] = list(stack)
        stack.pop()
        return

    # Get the routes in the case the condition is exists
    stack.append(int(current_node.Yes.values[0].split('-')[1]))
    _root_to_leaf_route(df, stack, routes)

    # Get the routes in the case the condition is not exists
    stack.append(int(current_node.No.values[0].split('-')[1]))
    _root_to_leaf_route(df, stack, routes)

    stack.pop()


def extract_nodes_samples(X, model_obj):
    """
        Extract the distribution of the samples in X inside the nodes of model_obj
    :param X:
    :param model_obj:
    :return:
    """
    # Extract all of the different routes in the trees
    forest_routes = extract_root_to_leaf_routes_for_forest(model_obj)

    # structure to collect which sample is shown in which node
    node_to_samples = {node: [] for node in model_obj.get_booster().trees_to_dataframe().ID}

    # check to which leaf each samples in each tree is reaching
    nodes = model_obj.apply(X)
    # if there is only one estimator
    if type(nodes) == np.int32:
        nodes = [nodes]

    for i in tqdm(range(X.shape[0]), total=X.shape[0], desc="Calculating samples distribution in trees"):
        sample_leafs = nodes[i]

        # Add the sample to each node in the relevant route
        # Changed this for multiclass option.. need to further investigate! TODO: keep investigate
        # for l in range(model_obj.n_estimators):
        for l in range(len(model_obj.get_booster().get_dump())):
            for n in forest_routes[l][sample_leafs[l]]:
                node_to_samples[f'{l}-{n}'].append(i)

    return node_to_samples
