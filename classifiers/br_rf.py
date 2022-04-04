from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import BinaryRelevance
import numpy as np
from evaluate import evaluate


def br_rf(X_train, X_test, y_train, y_test):
    classifier = BinaryRelevance(
        # classifier=RandomForestClassifier(),
        classifier=RandomForestClassifier(n_estimators=200),
        require_dense=[False, True]
    )
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    y_pred_prob_temp = classifier.predict_proba(X_test)
    y_pred_prob = np.zeros(y_pred_prob_temp.shape)
    for i in range(y_pred_prob_temp.shape[0]):
        y_pred_prob[i] = y_pred_prob_temp[i].toarray()

    return evaluate(predictions, y_test, y_pred_prob)


# if __name__ == '__main__':
#     data = pd.read_csv("../concatenated_features.csv", header=None).values
#     label = pd.read_csv("../lncRNA_label.csv").values
#     num = 0
#     result = 0
#     for i in range(10, 200, 10):
#         X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0)
#         average_pre_score, ham_loss, zero_one_loss_1, \
#         acc_score, jaccard_index = br_rf(X_train, X_test, y_train, y_test, i)
#         print(average_pre_score)
#         if average_pre_score > result:
#             result = average_pre_score
#             num = i
#     print(num, result)