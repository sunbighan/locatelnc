from sklearn.svm import SVC
from skmultilearn.problem_transform import BinaryRelevance
import numpy as np
from evaluate import evaluate


def br_svm(X_train, X_test, y_train, y_test):
    classifier = BinaryRelevance(
        classifier=SVC(probability=True),
        # classifier=SVC(probability=True),
        require_dense=[False, True]
    )
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    y_pred_prob_temp = classifier.predict_proba(X_test)
    y_pred_prob = np.zeros(y_pred_prob_temp.shape)
    for i in range(y_pred_prob_temp.shape[0]):
        y_pred_prob[i] = y_pred_prob_temp[i].toarray()

    return evaluate(predictions, y_test, y_pred_prob)