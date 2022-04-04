from sklearn.metrics import hamming_loss
from skmultilearn.adapt import MLARAM

from evaluate import evaluate
import numpy as np


def mlaram(X_train, X_test, y_train, y_test):
    classifier = MLARAM()  # train
    classifier.fit(X_train, y_train)
    # predict
    predictions = classifier.predict(X_test)
    y_pred_prob_temp = classifier.predict_proba(X_test)
    y_pred_prob = np.zeros(y_pred_prob_temp.shape)
    for i in range(y_pred_prob_temp.shape[0]):
        y_pred_prob[i] = y_pred_prob_temp[i]

    return evaluate(predictions, y_test, y_pred_prob)


