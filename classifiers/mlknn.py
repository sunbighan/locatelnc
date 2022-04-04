from sklearn.metrics import accuracy_score, hamming_loss
from skmultilearn.adapt import MLkNN
import numpy as np
from evaluate import evaluate


def mlknn(X_train, X_test, y_train, y_test):
    # params = {
    #     "k": [5, 8, 10, 12, 15, 17, 20, 30, 50, 100],
    #     's': [0.5, 0.75, 1, 1.25, 1.5]
    # }
    # cur_max = 0
    # temp = []
    # for k in params["k"]:
    #     for s in params["s"]:
    #         classifier = MLkNN(k=k, s=s)
    #         classifier.fit(X_train, y_train)
    #         # predict
    #         predictions = classifier.predict(X_test)
    #
    #         z = predictions.toarray()
    #         # print(z)
    #         accuracy = accuracy_score(y_test, z)
    #         y_pred_prob_temp = classifier.predict_proba(X_test)
    #         # print(y_pred_prob_temp.shape)
    #         y_pred_prob = np.zeros(y_pred_prob_temp.shape)
    #         for i in range(y_pred_prob_temp.shape[0]):
    #             y_pred_prob[i] = y_pred_prob_temp[i].toarray()
    #         average_pre_score, ham_loss, \
    #             zero_one_loss_1, acc_score, \
    #             jaccard_index = evaluate(predictions, y_test, y_pred_prob)
    #         if average_pre_score > cur_max:
    #             cur_max = average_pre_score
    #             temp.clear()
    #             temp.append(k)
    #             temp.append(s)
    # print(temp, cur_max)

    classifier = MLkNN(k=20, s=1.25)
    # classifier = MLkNN()
    # train
    classifier.fit(X_train, y_train)
    # predict
    predictions = classifier.predict(X_test)

    z = predictions.toarray()
    # print(z)
    accuracy = accuracy_score(y_test, z)
    y_pred_prob_temp = classifier.predict_proba(X_test)
    # print(y_pred_prob_temp.shape)
    y_pred_prob = np.zeros(y_pred_prob_temp.shape)
    for i in range(y_pred_prob_temp.shape[0]):
        y_pred_prob[i] = y_pred_prob_temp[i].toarray()
    # print("Accuracy on test set: %.2f%%" % (accuracy * 100))
    # print("Hamming loss on test set: ", hamming_loss(y_test, z))
    return evaluate(predictions, y_test, y_pred_prob)
    # print("average_precision_score: ", average_precision_score(y_test, y_pred_prob.toarray()))
    # print("recall_score: ", recall_score(y_test, z, average='micro'))
    # print("jaccard_score: ", jaccard_score(y_test, z, average='micro'))
    # print("roc_auc_score: ", roc_auc_score(y_test, z))