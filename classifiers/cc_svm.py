from sklearn.metrics import hamming_loss
from sklearn.svm import SVC
from skmultilearn.problem_transform import ClassifierChain
from evaluate import evaluate
import numpy as np


def cc_svm(X_train, X_test, y_train, y_test):
    # params = {
    #     "c": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
    #     'gamma': [0.001, 0.0001]
    # }
    # cur_max = 0
    # temp = []
    # for c in params["c"]:
    #     for gamma in params["gamma"]:
    #         classifier = ClassifierChain(
    #             classifier=SVC(C=c, gamma=gamma, probability=True),
    #             require_dense=[False, True]
    #         )  # train
    #         classifier.fit(X_train, y_train)
    #         # predict
    #         predictions = classifier.predict(X_test)
    #         y_pred_prob_temp = classifier.predict_proba(X_test)
    #         y_pred_prob = np.zeros(y_pred_prob_temp.shape)
    #         for i in range(y_pred_prob_temp.shape[0]):
    #             y_pred_prob[i] = y_pred_prob_temp[i].toarray()
    #         average_pre_score, ham_loss, \
    #             zero_one_loss_1, acc_score, \
    #             jaccard_index = evaluate(predictions, y_test, y_pred_prob)
    #         if average_pre_score > cur_max:
    #             cur_max = average_pre_score
    #             temp.clear()
    #             temp.append(c)
    #             temp.append(gamma)
    # print(temp, cur_max)

    classifier = ClassifierChain(
        classifier=SVC(probability=True),
        # classifier=SVC(probability=True),
        require_dense=[False, True]
    )    # train
    classifier.fit(X_train, y_train)
    # predict
    predictions = classifier.predict(X_test)
    y_pred_prob_temp = classifier.predict_proba(X_test)
    y_pred_prob = np.zeros(y_pred_prob_temp.shape)
    for i in range(y_pred_prob_temp.shape[0]):
        y_pred_prob[i] = y_pred_prob_temp[i].toarray()

    return evaluate(predictions, y_test, y_pred_prob)


