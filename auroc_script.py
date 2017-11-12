import utils, models_preprocessing, metrics
from sklearn.model_selection import KFold, train_test_split
import numpy
import matplotlib.pyplot as plt

input_shape = (101, 101, 4)
regs= [0, 0.3]

auroc_Graph = metrics.aurocGraph
confusionMatrix = metrics.confusionMatrix
# X_FILE = 'C:/Users/thuanhsi/Downloads/Strong_Lens_Code_daniel/flipped.npy'
# Y_FILE = 'C:/Users/thuanhsi/Downloads/Strong_Lens_Code_daniel/flipped_labels.npy'

print('loading data ...')
data = numpy.load(X_FILE)
labels = numpy.load(Y_FILE)
print('done!')

i = 0
records = []
fprs = []
tprs = []
thresholds = []
names = []
for reg in regs:
	model = models_preprocessing.compiledRegularizedConvnet(input_shape, reg) 

	Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, labels, test_size=0.3)

	trained_model = utils.train(model, 26, Xtrain, Ytrain, None, None, True)

	record = utils.test(trained_model, [confusionMatrix, auroc_Graph], Xtest, Ytest)
	
	if len(record)==4:
		names.append(record[0])
		fprs.append(record[1])
		tprs.append(record[2])
		thresholds.append(record[3])
	elif len(record)==1:
		confusionMatrix = record
	i += 1


line0, = plt.plot(fprs[0], tprs[0], linewidth='2')
line1, = plt.plot(fprs[1], tprs[1], linewidth='2')

plt.legend([line0, (line0, line1)], ["reg={0}".format(regs[0]), "reg={0}".format(regs[1])] , loc=2)
plt.title(names[0] + ' ROC curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
save_file = "./combined_auc.png"
plt.savefig(save_file)
plt.show()