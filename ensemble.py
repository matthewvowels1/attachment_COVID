import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from super_learner import SuperLearner
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.dummy import DummyClassifier
from tqdm import tqdm



def init_super_dict(output_type):
    est_list = [('EEC', EasyEnsembleClassifier()), ('BBC', BalancedBaggingClassifier()),
               ('RUS', RUSBoostClassifier()), ('BRF', BalancedRandomForestClassifier())]
#     est_list = [('LR', LogisticRegression(max_iter=1000))]

    return est_list




def classify(df_x, df_y, sl_list, classifier='ensemble'):
    # classifier == 'ensemble', or 'dummy'
    # start k-fold loop
    kf = KFold(n_splits=10)
    kf.get_n_splits(df_x)

    y_true = []
    y_preds = []
    x_tests = []
    for train_index, test_index in kf.split(df_x):

        if classifier == 'ensemble':
            clf = VotingClassifier(estimators=sl_list)
        elif classifier == 'dummy':
            clf = DummyClassifier(strategy='uniform')

        X_train, X_test = df_x[train_index], df_x[test_index]
        y_train, y_test = df_y[train_index], df_y[test_index]

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        y_true.append(y_test)
        y_preds.append(preds)
        x_tests.append(X_test)

    y_true = np.concatenate(y_true)
    y_preds = np.concatenate(y_preds)
    x_tests = np.concatenate(x_tests)
    return y_true, y_preds


def get_results(y_true, y_preds, sample_size):
    num_runs = len(y_true) // sample_size
    if num_runs == 1:
        prename = 'classification_results/ensemble_'
        cm = confusion_matrix(y_true, y_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(prename+'conf_mat_unnormalized.png')
        plt.close()
    else:
        prename = 'classification_results/dummy'


    cm_recalls = []
    cm_precs = []
    for i in tqdm(range(num_runs)):
        cm = confusion_matrix(y_true[i * sample_size:i * sample_size + sample_size],
                              y_preds[i * sample_size:i * sample_size + sample_size], normalize='true')
        cm_recalls.append(cm)
        cm = confusion_matrix(y_true[i * sample_size:i * sample_size + sample_size],
                              y_preds[i * sample_size:i * sample_size + sample_size], normalize='pred')
        cm_precs.append(cm)

    cm_recalls = np.asarray(cm_recalls)
    cm_precs = np.asarray(cm_precs)
    cm_recall = cm_recalls.mean(0)
    cm_prec = cm_precs.mean(0)

    # Since each row represents the total number of actual values for each class label,
    # the final normalized matrix will show us the percentage ie. out of all true labels
    # for a particular class, what was the % prediction of each class made by our model
    # for that specific true label.

    # Each row is all true values for a single class, and the numbers tell us what percent of all the true labels
    # for each class were correctly predicted.

    print('Diag is recall')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_recall)
    disp.plot()
    plt.savefig(prename + 'conf_mat_normed_rows_recall.png')
    plt.close()

    if num_runs > 10:
        cm_recall_upper = cm_recall + 1.96 * cm_recalls.std(0)
        print(cm_recall_upper.shape)
        print('upper 1.96 CI bound')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_recall_upper)
        disp.plot()
        plt.savefig(prename + 'conf_mat_normed_rows_recall_upper_CI.png')
        plt.close()

    # Each column is all predicted values for a single class, and the numbers tell us what percent of all the predicted
    # labels for each class were correct.
    print('Diag is precision')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_prec)
    disp.plot()
    plt.savefig(prename + 'conf_mat_normed_cols_prec.png')
    plt.close()

    if num_runs > 10:
        cm_prec_upper = cm_prec + 1.96 * cm_precs.std(0)
        print('upper 1.96 CI bound')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_prec_upper)
        disp.plot()
        plt.savefig(prename + 'conf_mat_normed_cols_prec_upper_CI.png')
        plt.close()

    print(classification_report_imbalanced(y_true, y_preds))



if __name__ == "__main__":

    n_bootstraps = 1000
    sl_list = init_super_dict('categorical')

    #data_filename = 'W3_obesity_short_use.csv'  # v1
    data_filename = 'W3_obesity_short_use_v2.csv'  # v2
    codebook_stuff = '220227codebook.csv'

    data = pd.read_csv(data_filename, na_values=[' ', -99, 'NA', '#VALUE!'])
    codebook = pd.read_csv(os.path.join(codebook_stuff))
    cols = data.columns
    print(len(data))

    # check for nans (listwise delete for now...)
    data.dropna(inplace=True)
    nan_cols = []

    print(len(data))

    # turn sum scores into averages using codebook information

    for divvo in codebook.values:
        col_name = divvo[0]
        divider = divvo[1]
        data[col_name] = data[col_name] / divider

    outcome_col = 'W3_Weight_Category'
    pred_cols = set(cols) - set([outcome_col])
    df_x = np.asarray(data[pred_cols].values)
    df_y = np.asarray(data[outcome_col].values)
    df_x = (df_x - df_x.mean()) / df_x.std()

    y_true, y_preds = classify(df_x, df_y, classifier='ensemble', sl_list=sl_list)

    y_true_dummys = []
    y_preds_dummys = []
    for i in range(n_bootstraps):
        y_true_d, y_preds_d = classify(df_x, df_y, classifier='dummy', sl_list=None)
        y_true_dummys.append(y_true_d)
        y_preds_dummys.append(y_preds_d)

    y_true_dummys = np.concatenate(y_true_dummys)
    y_preds_dummys = np.concatenate(y_preds_dummys)


    print(np.unique(y_true, return_counts=True))
    print(np.unique(y_preds, return_counts=True))
    print(np.unique(y_true_dummys, return_counts=True))
    print(np.unique(y_preds_dummys, return_counts=True))

    get_results(y_true=y_true, y_preds=y_preds, sample_size=len(y_true))
    get_results(y_true=y_true_dummys, y_preds=y_preds_dummys, sample_size=len(y_true))