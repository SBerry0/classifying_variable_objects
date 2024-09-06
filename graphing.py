import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support, RocCurveDisplay, roc_curve, auc
from itertools import cycle

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize

def graph_features(feature_importances):
    all_features = ('num_selected_g_fov,mean_obs_time_g_fov,time_duration_g_fov,min_mag_g_fov,max_mag_g_fov,'
                    'mean_mag_g_fov,median_mag_g_fov,range_mag_g_fov,trimmed_range_mag_g_fov,std_dev_mag_g_fov,'
                    'skewness_mag_g_fov,kurtosis_mag_g_fov,mad_mag_g_fov,abbe_mag_g_fov,iqr_mag_g_fov,'
                    'stetson_mag_g_fov,std_dev_over_rms_err_mag_g_fov,outlier_median_g_fov,num_selected_bp,'
                    'mean_obs_time_bp,time_duration_bp,min_mag_bp,max_mag_bp,mean_mag_bp,median_mag_bp,range_mag_bp,'
                    'trimmed_range_mag_bp,std_dev_mag_bp,skewness_mag_bp,kurtosis_mag_bp,mad_mag_bp,abbe_mag_bp,'
                    'iqr_mag_bp,stetson_mag_bp,std_dev_over_rms_err_mag_bp,outlier_median_bp,num_selected_rp,'
                    'mean_obs_time_rp,time_duration_rp,min_mag_rp,max_mag_rp,mean_mag_rp,median_mag_rp,range_mag_rp,'
                    'trimmed_range_mag_rp,std_dev_mag_rp,skewness_mag_rp,kurtosis_mag_rp,mad_mag_rp,abbe_mag_rp,'
                    'iqr_mag_rp,stetson_mag_rp,std_dev_over_rms_err_mag_rp,outlier_median_rp')
    feature_names = all_features.split(',')

    feat_importances = pd.Series(feature_importances, index=feature_names)
    feat_importances = feat_importances.sort_values()
    sns.barplot(feat_importances, orient='y')

    # ax = feat_importances.plot(kind='barh')
    # ax.invert_yaxis()
    #
    # ax.figure(figsize=(10, 6))
    plt.show()

    # plt.title('Feature Importances')
    # plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    # plt.yticks(range(len(indices)), [features[i] for i in indices])
    # plt.xlabel('Relative Importance')
    # plt.show()


def plot_acc(test_acc_hist, train_acc_hist):
    plt.plot(train_acc_hist, label="train")
    plt.plot(test_acc_hist, label="test")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


def plot_loss(test_loss_hist, train_loss_hist):
    # Plot the loss and accuracy
    plt.plot(train_loss_hist, label="train")
    plt.plot(test_loss_hist, label="test")
    plt.xlabel("epochs")
    plt.ylabel("cross entropy")
    plt.legend()
    plt.show()

def graph_confusion_matrix(classes, y_pred, y_test):
    print("type", type(y_pred), "y_pred", y_pred)
    print("type", type(y_test), "y_pred", y_test)
    cm = confusion_matrix(y_test.detach(), y_pred, labels=classes)
    cm_true = confusion_matrix(y_test, y_pred, labels=classes, normalize='true')
    cm_pred = confusion_matrix(y_test, y_pred, labels=classes, normalize='pred')

    sns.heatmap(cm, annot=True, fmt='.8f', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Absolute Confusion Matrix')
    plt.show()

    sns.heatmap(cm_true, annot=True, fmt='.8f', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('True Normalized Confusion Matrix')
    plt.show()

    sns.heatmap(cm_pred, annot=True, fmt='.8f', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Prediction Normalized Confusion Matrix')
    plt.show()


def autopct_format(pct, values):
    total = sum(values)
    val = int(round(pct * total / 100.0))
    # Specify slices where you want the text outside
# if index in ['OTHER', 'RR', 'S']:  # Text outside for slices 'A' and 'C'
    return f'{val:,}\n({pct:.1f}%)'

def autopct(values):
    def format(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        # return f'{val:,} ({pct:.1f}%)'
        return f'{val:,})'
    return format

def autolabel(indices):
    pass
    # def

def graph_data_types(data):
    data_types = data.value_counts('best_class_name')
    counts = [data_types]
    print(data_types)
    # plt.figure(figsize=(10, 6))
    # print(data_types)
    # explode = (0.05,) * len(data_types)
    # data_types.plot(kind='pie',
    #                 autopct=autopct(data_types),
    #                 textprops={'fontsize': 10},
    #                 pctdistance=0.5)
    # ax.title("Data Types By Percentage")

    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(data_types,
                                      autopct=lambda pct: autopct_format(pct, data_types),
                                      textprops={'fontsize': 9},
                                      # labels=data_types.index,
                                      pctdistance=1.2)  # Distance of the percentage from center

    # Set labels outside the slices
    # for text in autotexts:
    #     text.set_fontsize(10)
    #     text.set_color('black')


    # Add a legend to the side
    ax.legend(wedges, data_types.index, title="Classes", loc="center left", bbox_to_anchor=(1.25, 0, 0.5, 1))
    plt.show()


def plot_precision_recall_curve(y_score, y_test, classes):
    n_classes = len(classes)
    from sklearn.metrics import average_precision_score, precision_recall_curve

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    print(y_score.shape)
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_test.ravel(), y_score.ravel()
    )
    average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")

    # setup plot details
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "darkviolet", "red", "teal", "salmon", "darkolivegreen", "mediumvioletred"])

    _, ax = plt.subplots(figsize=(7, 8))

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

    for i, color in zip(range(n_classes), colors):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"Precision-recall for class {classes[i]}", color=color)

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.legend(handles=handles, labels=labels, loc="best")
    # ax.set_title("Extension of Precision-Recall curve to multi-class")

    plt.show()


def plot_roc_curve(y_score, y_train, y_test, class_names, n_classes=9):
    label_binarizer = LabelBinarizer().fit(y_train)
    print(y_test.shape)
    y_onehot_test = label_binarizer.transform(y_test)
    fpr, tpr, roc_auc = dict(), dict(), dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)
    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
    # Average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.5f}")
    # %%
    # This computation is equivalent to simply calling
    macro_roc_auc_ovr = roc_auc_score(
        y_test,
        y_score,
        multi_class="ovr",
        average="macro",
    )
    print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{macro_roc_auc_ovr:.5f}")
    # %%
    # Plot all OvR ROC curves together
    # --------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.4f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.4f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "darkviolet", "red", "teal", "salmon", "darkolivegreen", "mediumvioletred"])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"ROC curve for {class_names[class_id]}",
            color=color,
            ax=ax,
            plot_chance_level=(class_id == 2),
        )
    _ = ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        # title="Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass",
    )
    plt.show()



def roc_scoring(rfm_score_classes, rfm_y_score, y_test):
    print('OVO:')
    print('macro:', roc_auc_score(y_test, rfm_y_score, multi_class='ovo', labels=rfm_score_classes, average='macro'))
    print('weighted:',
          roc_auc_score(y_test, rfm_y_score, multi_class='ovo', labels=rfm_score_classes, average='weighted'))

    print('OVR:')
    print('micro:',
          roc_auc_score(y_test, rfm_y_score, multi_class='ovr', labels=rfm_score_classes, average='micro'))
    print('macro:',
          roc_auc_score(y_test, rfm_y_score, multi_class='ovr', labels=rfm_score_classes, average='macro'))
    print('weighted:',
          roc_auc_score(y_test, rfm_y_score, multi_class='ovr', labels=rfm_score_classes, average='weighted'))
    nones = roc_auc_score(y_test, rfm_y_score, multi_class='ovr', labels=rfm_score_classes, average=None)
    roc_scores = pd.Series(nones, rfm_score_classes)
    print(roc_scores)


def precision_recall_scoring(rfm_score_classes, y_pred, y_test):
    print("\nPrecision Recall Scoring:")
    precision, recall, fbeta, support = precision_recall_fscore_support(y_test, y_pred, labels=rfm_score_classes,
                                                                        average='micro')
    print(f'Micro: precision: {precision}, recall: {recall}, fbeta: {fbeta}, support: {support}')
    precision, recall, fbeta, support = precision_recall_fscore_support(y_test, y_pred, labels=rfm_score_classes,
                                                                        average='macro')
    print(f'Macro: precision: {precision}, recall: {recall}, fbeta: {fbeta}, support: {support}')
    precision, recall, fbeta, support = precision_recall_fscore_support(y_test, y_pred, labels=rfm_score_classes,
                                                                        average='weighted')
    print(f'Weighted: precision: {precision}, recall: {recall}, fbeta: {fbeta}, support: {support}')

    precision, recall, fbeta, support = precision_recall_fscore_support(y_test, y_pred, labels=rfm_score_classes,
                                                                        average=None)
    print(f'None: precision: {precision}, recall: {recall}, fbeta: {fbeta}, support: {support}')
    precision_scores = pd.Series(precision, rfm_score_classes)
    recall_scores = pd.Series(recall, rfm_score_classes)
    print(precision_scores)
    print(recall_scores)
