import sklearn.metrics as metrics
from scripts.bayesianML import *
fulldata = np.loadtxt("data/iris.txt",delimiter=",")
print("data read")
inputs = fulldata[:,:-1]
targets = fulldata[:,-1]
predict_labels = perform_leave_one_out(inputs,targets)
acc = metrics.accuracy_score(targets,predict_labels)
perf_metrics = metrics.precision_recall_fscore_support(targets,predict_labels)

precision = perf_metrics[0]
recall = perf_metrics[1]

print("Metrics For first Part")
print("-"*30)
print("acc",acc)
print("recall",recall)
print("precision",precision)
print("-"*30)
print("")
print("")
