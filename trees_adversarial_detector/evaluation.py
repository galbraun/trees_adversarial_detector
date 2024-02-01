import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from bokeh.io import save, output_file
from bokeh.layouts import row, widgetbox, layout
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, TapTool, OpenURL
from bokeh.models import HoverTool, LinearColorMapper, ColorBar
from bokeh.models import HoverTool, LinearColorMapper, ColorBar
from bokeh.models.widgets import Select, DataTable, TableColumn, Tabs, Panel
from bokeh.palettes import Viridis11
from bokeh.palettes import Viridis11
from bokeh.plotting import figure
from bokeh.transform import transform
from bokeh.transform import transform
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import homogeneity_score, completeness_score
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import faiss

sns.set()


def evaluate_experiment():
    pass


#https://towardsdatascience.com/make-knn-300-times-faster-than-scikit-learns-in-20-lines-5e29d74e76bb
import numpy as np
import faiss


# class FaissKNeighbors:
#     def __init__(self, k=5):
#         self.index = None
#         self.y = None
#         self.k = k
#
#     def fit(self, X, y):
#         self.index = faiss.IndexFlatL2(X.shape[1])
#         self.index.add(X.astype(np.float32))
#         self.y = y
#
#     def predict(self, X):
#         distances, indices = self.index.search(X.astype(np.float32), k=self.k)
#         votes = self.y[indices]
#         predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
#         return predictions

#https://github.com/shankarpm/faiss_knn/blob/master/FAISS_KNN_Notebook.ipynb
from scipy import stats

class FaissKNeighbors:

    def __init__(self, k, faiss):
        self.k = k  # k nearest neighbor value
        self.faissIns = faiss  # FAISS instance
        self.index = 0
        self.gpu_index_flat = 0
        self.train_labels = []
        self.test_label_faiss_output = []

    def fit(self, train_features, train_labels):
        self.train_labels = train_labels
        self.index = self.faissIns.IndexFlatL2(train_features.shape[1])  # build the index
        self.index.add(train_features)  # add vectors to the index

    # def fit(self, train_features, train_labels):
    #     no_of_gpus = self.faissIns.get_num_gpus()
    #     self.train_labels = train_labels
    #     self.gpu_index_flat = self.index = self.faissIns.IndexFlatL2(train_features.shape[1])  # build the index
    #     if no_of_gpus > 0:
    #         self.gpu_index_flat = self.faissIns.index_cpu_to_all_gpus(self.index)
    #
    #     self.gpu_index_flat.add(train_features)  # add vectors to the index
    #     return no_of_gpus

    def predict(self, test_features):
        distance, test_features_faiss_Index = self.index.search(test_features, self.k)
        self.test_label_faiss_output = stats.mode(self.train_labels[test_features_faiss_Index], axis=1)[0]
        self.test_label_faiss_output = np.array(self.test_label_faiss_output.ravel())
        # for test_index in range(0,test_features.shape[0]):
        #    self.test_label_faiss_output[test_index] = stats.mode(self.train_labels[test_features_faiss_Index[test_index]])[0][0] #Counter(self.train_labels[test_features_faiss_Index[test_index]]).most_common(1)[0][0]
        return self.test_label_faiss_output

    # def predit(self, test_features):
    #     distance, test_features_faiss_Index = self.gpu_index_flat.search(test_features, self.k)
    #     self.test_label_faiss_output = stats.mode(self.train_labels[test_features_faiss_Index], axis=1)[0]
    #     self.test_label_faiss_output = np.array(self.test_label_faiss_output.ravel())
    #     return self.test_label_faiss_output

    def getAccuracy(self, test_labels):
        accuracy = (self.test_label_faiss_output == test_labels).mean()
        return round(accuracy, 2)


def evaluate_embedding(X, X_embedded, y, train_ix, test_ix):
    embedding_acc = []

    original_acc = []

    for i in tqdm(range(2, 21), total=19, desc="Checking KNN for both feature spaces for different Ks"):
        neigh = FaissKNeighbors(k=i, faiss=faiss)
        neigh.fit(X_embedded[train_ix].astype('float32'), y[train_ix].astype('float32'))
        # embedding_auc.append(roc_auc_score(y, neigh.predict(X_embedded)))
        embedding_acc.append(accuracy_score(y[test_ix], neigh.predict(X_embedded[test_ix].astype('float32'))))

        neigh.fit(X[train_ix].astype('float32'), y[train_ix].astype('float32'))
        # neigh.fit(X.iloc[train_ix], y[train_ix])

        # original_auc.append(roc_auc_score(y, neigh.predict(X)))
        original_acc.append(accuracy_score(y[test_ix], neigh.predict(X[test_ix].astype('float32'))))
        # original_acc.append(accuracy_score(y[test_ix], neigh.predict(X.iloc[test_ix])))

    # return embedding_auc, embedding_acc, original_auc, original_acc
    return embedding_acc, original_acc


def compare_viz(original, embedded, metric_type, dataset):
    t = plt.plot(range(2, 21), original, label=f'original feature space (avg = {np.mean(original):.3f})')
    t = plt.plot(range(2, 21), embedded, label=f'embedding space (avg = {np.mean(embedded):.3f})')
    # plt.gca().set_ylim([0.5,1.2])
    plt.xticks(list(range(2, 21, 2)))
    plt.legend()

    plt.title(f'KNN {metric_type} compare over {dataset} dataset with XGBoost')
    plt.xlabel('K neighbors')
    plt.ylabel(f'{metric_type}')


def dim_reduction(X, perplexity=5):
    return TSNE(perplexity=perplexity, verbose=2, random_state=82).fit_transform(X)


def generate_2d_bokeh(X, y, model_predictions, title='Some graph'):
    true_positive_indices = np.where((y == 1) & (model_predictions == True))
    true_negative_indices = np.where((y == 0) & (model_predictions == False))
    false_positive_indices = np.where((y == 0) & (model_predictions == True))
    false_negative_indices = np.where((y == 1) & (model_predictions == False))

    column_data_true_positive = ColumnDataSource(
        {'index': true_negative_indices[0], 'x': X[true_positive_indices, 0], 'y': X[true_positive_indices, 1],
         'label': y[true_positive_indices], 'prediction': model_predictions[true_positive_indices]})
    column_data_true_negative = ColumnDataSource(
        {'index': true_negative_indices[0], 'x': X[true_negative_indices, 0], 'y': X[true_negative_indices, 1],
         'label': y[true_negative_indices], 'prediction': model_predictions[true_negative_indices]})
    column_data_false_positive = ColumnDataSource(
        {'index': false_positive_indices[0], 'x': X[false_positive_indices, 0], 'y': X[false_positive_indices, 1],
         'label': y[false_positive_indices], 'prediction': model_predictions[false_positive_indices]})
    column_data_false_negative = ColumnDataSource(
        {'index': false_negative_indices[0], 'x': X[false_negative_indices, 0], 'y': X[false_negative_indices, 1],
         'label': y[false_negative_indices], 'prediction': model_predictions[false_negative_indices]})

    hover_tool = HoverTool()
    hover_tool.tooltips = [
        ("label", "@label"),
        ("prediction", "@prediction"),
        ("index", "@index")
    ]

    fig = figure(title=title, width=1000, height=1000)
    fig.scatter('x', 'y', source=column_data_true_positive, size=7, color='red', legend='TP')
    fig.scatter('x', 'y', source=column_data_true_negative, size=7, color='green', legend='TN')
    fig.scatter('x', 'y', source=column_data_false_positive, size=7, color='blue', legend='FP')
    fig.scatter('x', 'y', source=column_data_false_negative, size=7, color='purple', legend='FN')
    fig.legend.click_policy = "hide"

    fig.add_tools(hover_tool)
    return layout([[fig]], sizing_mode='fixed')


def generate_2d_bokeh_nodes(X, feat_index, title='Some graph'):
    column_data = ColumnDataSource(
        {'index': np.arange(X.shape[0]), 'x': X[:, 0], 'y': X[:, 1], 'feat_index': feat_index})

    hover_tool = HoverTool()
    hover_tool.tooltips = [
        ("index", "@index"),
        ("feat_index", "@feat_index")
    ]

    fig = figure(title=title, width=1000, height=1000)
    fig.scatter('x', 'y', source=column_data, size=7, color='red')
    fig.legend.click_policy = "hide"

    fig.add_tools(hover_tool)
    return layout([[fig]], sizing_mode='fixed')


def dbscan_clustering_evaluate(X, y, eps=(0.1, 0.5, 1, 5, 10, 20, 30, 100, 250, 500, 1000),
                               min_samples=(2, 3, 5, 7, 10, 15)):
    with tqdm(total=len(eps) * len(min_samples), desc='calculating dbscans') as pbar:
        completeness_results = []
        homogeneity_results = []
        for ep in eps:
            completness_line = []
            homogeneity_line = []
            for min_s in min_samples:
                clustering = DBSCAN(eps=ep, min_samples=min_s, n_jobs=-1, metric='cosine').fit(X)
                completness_line.append(completeness_score(y, clustering.labels_))
                homogeneity_line.append(homogeneity_score(y, clustering.labels_))
                pbar.update(1)
            completeness_results.append(completness_line)
            homogeneity_results.append(homogeneity_line)
    return np.array(completeness_results), np.array(homogeneity_results), eps, min_samples


def dbscan_visualization(X, y, dbscan_labels, title='DBSCAN viz'):
    clusters = set(dbscan_labels)
    clusters_indices = {index: {} for index in clusters}
    for cl_id in clusters:
        clusters_indices[cl_id]['indices'] = np.where(dbscan_labels == cl_id)

    for cl_id in clusters:
        indices = clusters_indices[cl_id]['indices']
        clusters_indices[cl_id]['columnDataSource'] = ColumnDataSource(
            {'x': X[indices, 0], 'y': X[indices, 1], 'label': y[indices], 'cluster': dbscan_labels[indices]})

    hover_tool = HoverTool()
    hover_tool.tooltips = [
        ("cluster", "@cluster"),
    ]

    fig = figure(title=title, width=1000, height=1000)
    for cl_id in clusters:
        if cl_id == -1:
            fig.scatter('x', 'y', source=clusters_indices[cl_id]['columnDataSource'], size=7, color='red',
                        legend_label=str(cl_id))
        else:
            fig.scatter('x', 'y', source=clusters_indices[cl_id]['columnDataSource'], size=7, color='green',
                        legend_label=str(cl_id))

    fig.legend.click_policy = "hide"
    fig.add_tools(hover_tool)
    return layout([[fig]], sizing_mode='fixed')


def multiclass_visualization(X, y, title='multiclass viz'):
    colors = ['red', 'blue', 'yellow', 'orange', 'purple', 'black', 'grey']
    labels = set(y)
    labels_indices = {index: {} for index in labels}
    for y_id in labels:
        labels_indices[y_id]['indices'] = np.where(y == y_id)

    for y_id in labels:
        indices = labels_indices[y_id]['indices']
        labels_indices[y_id]['columnDataSource'] = ColumnDataSource(
            {'x': X[indices, 0], 'y': X[indices, 1], 'idx': indices[0], 'label': y[indices]})

    hover_tool = HoverTool()
    hover_tool.tooltips = [
        ("label", "@label"),
        ("idx", "@idx")
    ]

    fig = figure(title=title, width=1000, height=1000)
    for y_id in labels:
        fig.scatter('x', 'y', source=labels_indices[y_id]['columnDataSource'], size=7, color=colors[y_id],
                    legend_label=str(y_id))

    fig.legend.click_policy = "hide"
    fig.add_tools(hover_tool)
    return layout([[fig]], sizing_mode='fixed')
