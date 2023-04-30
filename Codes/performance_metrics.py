from sklearn.metrics import make_scorer, confusion_matrix, roc_curve, roc_auc_score, auc, precision_recall_curve, roc_auc_score, average_precision_score, accuracy_score, recall_score, precision_score

# AUROC
auroc = roc_auc_score(y_test, predictions, multi_class="ovr")
print(f"AUROC: {auroc}")

# AUPR
aupr = average_precision_score(y_test, predictions)
print(f"AUPR: {aupr}")

# Accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy}")

# Sensitivity (Recall)
sensitivity = recall_score(true_labels, predicted_labels, average="weighted")
print(f"Sensitivity: {sensitivity}")

# Specificity
cm = confusion_matrix(true_labels, predicted_labels)
specificity = cm.diagonal() / cm.sum(axis=1)
specificity = np.mean(specificity)  # average specificity
print(f"Specificity: {specificity}")

# Recall
recall = recall_score(true_labels, predicted_labels, average="weighted")
print(f"Recall: {recall}")

# Precision
precision = precision_score(true_labels, predicted_labels, average="weighted")
print(f"Precision: {precision}")

f1 = (2 * precision * recall)/(precision + recall)
print(f"F1: {f1}")


# Binarize the labels
y_test_binarized = label_binarize(y_test, classes=[0,1,2])
n_classes = y_test_binarized.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

# Compute PR curve and PR area for each class
precision = dict()
recall = dict()
pr_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], predictions[:, i])
    pr_auc[i] = average_precision_score(y_test_binarized[:, i], predictions[:, i])

# Compute macro-average ROC curve and ROC area
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes
roc_auc["macro"] = auc(all_fpr, mean_tpr)

# Interpolate precision and recall curves to have the same length
recall_grid = np.linspace(0, 1, 1000)
interp_precision = np.zeros((n_classes, recall_grid.size))

for i in range(n_classes):
    interp_precision[i, :] = np.interp(recall_grid, recall[i], precision[i])

mean_precision = np.mean(interp_precision, axis=0)

# Plot macro-averaged ROC curve
plt.plot(all_fpr, mean_tpr, label='Macro-averaged ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) for Multi-label Classification')
plt.legend(loc="lower right")
