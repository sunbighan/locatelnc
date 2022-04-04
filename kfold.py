import pandas as pd
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from classifiers.br_rf import br_rf
from classifiers.br_svm import br_svm
from classifiers.br_xgboost import br_xgboost
from classifiers.cc_svm import cc_svm
from classifiers.cc_rf import cc_rf
from classifiers.cc_xgboost import cc_xgboost
from classifiers.lp_rf import lp_rf
from classifiers.lp_svm import lp_svm
from classifiers.lp_xgboost import lp_xgboost
from classifiers.mlknn import mlknn
from classifiers.mlaram import mlaram
from classifiers.lift import mLIFT


def embedding(type):
    data = []
    label = []
    weights = [3.24476325968480, -1.86487400007822, -0.864571489921314, -1.61626539791651]

    if type == "kmer":
        data = pd.read_csv("lncRNA_Kmer4.csv")
        label = pd.read_csv("lncRNA_label.csv")
    elif type == "human":
        data = pd.read_csv("./concat/human_weighted_concat.csv", header=None)
        label = pd.read_csv("human_lncRNA_label.csv")
    elif type == "final":
        data = pd.read_csv("./concat/weighted_concat.csv", header=None)
        label = pd.read_csv("lncRNA_label.csv")
    elif type == "kmer1":
        data = pd.read_csv("lncRNA_Kmer1.csv", header=None)
        label = pd.read_csv("lncRNA_label.csv")
    elif type == "kmer2":
        data = pd.read_csv("lncRNA_Kmer2.csv", header=None)
        label = pd.read_csv("lncRNA_label.csv")
    elif type == "kmer3":
        data = pd.read_csv("lncRNA_Kmer3.csv", header=None)
        label = pd.read_csv("lncRNA_label.csv")
    elif type == "deeplncloc":
        data = pd.read_csv("1.5f.csv", header=None)
        label = pd.read_csv("lncRNA_label.csv")
    elif type == "psednc":
        data = pd.read_csv("./pse/l2w02.csv", header=None)
        label = pd.read_csv("lncRNA_label.csv")
    elif type == "bio":
        data = pd.read_csv("bio_feature.csv", header=None)
        label = pd.read_csv("lncRNA_label.csv")
    elif type == "new_bio":
        data = pd.read_csv("new_bio_feature.csv", header=None)
        label = pd.read_csv("lncRNA_label.csv")
    elif type == 'concat':
        data = pd.read_csv("./concat/direct_concat.csv", header=None)
        label = pd.read_csv("lncRNA_label.csv")
    elif type == 'concat2':
        data = pd.read_csv("concatenated_features.csv", header=None)
        label = pd.read_csv("lncRNA_label.csv")
    elif type == 'test0':
        data = pd.read_csv("ori_deep.csv", header=None)
        label = pd.read_csv("lncRNA_label.csv")
        print("原特征大小：", data.shape)
    elif type == 'test1':
        data = pd.read_csv("pca.csv", header=None)
        label = pd.read_csv("lncRNA_label.csv")
        print("pca后特征大小：", data.shape)
    elif type == '0.5f':
        data = pd.read_csv("0.5f.csv", header=None)
        label = pd.read_csv("lncRNA_label.csv")
        print("0.5f特征大小：", data.shape)
    elif type == '1.0f':
        data = pd.read_csv("1.0f.csv", header=None)
        label = pd.read_csv("lncRNA_label.csv")
        print("1.0f特征大小:", data.shape)
    elif type == '1.5f':
        data = pd.read_csv("1.5f.csv", header=None)
        label = pd.read_csv("lncRNA_label.csv")
        print("1.5f特征大小:", data.shape)
    elif type == '2.0f':
        data = pd.read_csv("1.0f.csv", header=None)
        label = pd.read_csv("lncRNA_label.csv")
        print("2.0f特征大小:", data.shape)
    X = data.values
    y = label.values
    if type == "deeplncloc":
        transfer = PCA(n_components=0.99)
        X = transfer.fit_transform(X)
    # if type == "concat":
    #     X[:, 0:256] = X[:, 0:256].dot(weights[0])
    #     X[:, 256:3655] = X[:, 256:3655].dot(weights[1])
    #     X[:, 3655:3761] = X[:, 3655:3761].dot(weights[2])
    #     X[:, 3761:3812] = X[:, 3761:3812].dot(weights[3])
    from sklearn.model_selection import KFold
    # 10-fold cross validation
    kf = KFold(n_splits=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    train_data = X_train
    train_label = y_train
    pd.DataFrame(train_data).to_csv("train_data.csv", header=None, index=None)
    pd.DataFrame(train_label).to_csv("train_label.csv", header=None, index=None)
    test_data = X_test
    test_label = y_test
    pd.DataFrame(test_data).to_csv("test_data.csv", header=None, index=None)
    pd.DataFrame(test_label).to_csv("test_label.csv", header=None, index=None)
    X_trains = []
    X_tests = []
    y_trains = []
    y_tests = []

    for train, test in kf.split(train_data):
        X_train, y_train = train_data[train], train_label[train]
        X_test, y_test = train_data[test], train_label[test]
        X_trains.append(X_train)
        y_trains.append(y_train)
        X_tests.append(X_test)
        y_tests.append(y_test)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # print("[Train set size]: ", X_train.shape)
    # print("[Test set size]: ", X_test.shape)
    return X_trains, X_tests, y_trains, y_tests


def convert():
    result = np.zeros((965, 106))
    with open("lncRNA_tab.csv") as file:
        count = 0
        for line in file:
            temp = line.split()
            result[count, :] = temp
            count += 1
    df = DataFrame(result)
    df.to_csv("psednc.csv", header=None, index=None)


def large_embedding(filename):
    data = pd.read_csv(filename, header=None)
    print(data.shape)
    label = pd.read_csv("lncRNA_label.csv")
    X = data.values
    y = label.values
    # 20%独立测试集 + 10折交叉验证
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    from sklearn.model_selection import KFold
    # 10-fold cross validation
    train_data = X_train
    train_label = y_train
    pd.DataFrame(train_data).to_csv("train_data.csv", header=None, index=None)
    pd.DataFrame(train_label).to_csv("train_label.csv", header=None, index=None)
    test_data = X_test
    test_label = y_test
    pd.DataFrame(test_data).to_csv("test_data.csv", header=None, index=None)
    pd.DataFrame(test_label).to_csv("test_label.csv", header=None, index=None)
    kf = KFold(n_splits=10)
    X_trains = []
    X_tests = []
    y_trains = []
    y_tests = []

    for train, test in kf.split(train_data):
        X_train, y_train = train_data[train], train_label[train]
        X_test, y_test = train_data[test], train_label[test]
        X_trains.append(X_train)
        y_trains.append(y_train)
        X_tests.append(X_test)
        y_tests.append(y_test)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # print("[Train set size]: ", X_train.shape)
    # print("[Test set size]: ", X_test.shape)
    return X_trains, X_tests, y_trains, y_tests


if __name__ == '__main__':
    # fold valid setting
    ap_result = {}
    hamming_loss_result = {}
    one_error_result = {}
    acc_score_result = {}
    ranking_loss_result = {}

    ap_m = []
    hamming_loss_m = []
    one_error_m = []
    acc_m = []
    ranking_loss_m = []

    evaluates = [ap_result, hamming_loss_result, one_error_result, acc_score_result, ranking_loss_result]

    # kmer:256, deep:774, bio:51, pse:106
    # embedded_datasets = ["kmer", "deeplncloc", "new_bio", "psednc", "concat"]
    # embedded_datasets = ["kmer1", "kmer2", "kmer3", "kmer"]
    # embedded_datasets = ["test0", "test1", "test2", "test3", "test4", "test5"]
    # embedded_datasets = ["kmer", "deeplncloc", "bio", "psednc"]
    embedded_datasets = ["final", "human"]
    # embedded_datasets = ["0.5f", "1.0f", "1.5f", "2.0f"]
    # embedded_datasets = ["kmer", "deeplncloc", "new_bio", "psednc"]
    # kmer参数
    kmer=[3, 4, 5]
    # lamada = [2, 3, 4]
    # 向量化维度
    # weights = [64, 128]
    # 子序列数量
    nums = [32, 64, 128]

    # for k in kmer:
    #     for w in weights:
    #         for n in nums:
    #             embedded_datasets.append("deep_data/" + str(k) + "mer_" + str(w) + "vec_" + str(n) + "sub_f.csv")
    # lamada = [3, 4]
    # weights = ["01", "02", "03", "04", "05"]
    # for l in lamada:
    #     for w in weights:
    #         embedded_datasets.append("pse/l" + str(l) + "w" + w + ".csv")
    # print(embedded_datasets)

    # classifiers = ["br_svm", "br_rf", "br_xgboost",
    #                "cc_svm", "cc_rf", "cc_xgboost",
    #                "lp_svm", "lp_rf", "lp_xgboost",
    #                "mlknn", "mlaram", "mLIFT"]

    classifiers = ["cc_xgboost"]
    # classifiers = ["lp_rf"]
    # classifiers = ["ovr_svm", "cc_svm", "br_svm", "lp_svm"]
    for method in evaluates:
        for classifier in classifiers:
            method[classifier] = 0
    print(evaluates)
    # test
    my_table = PrettyTable()
    my_table.field_names = ["Dataset", "Classifier", "AP", "Hamming Loss", "One Error", "ACC", "Ranking Loss"]
    # 此处修改fold数
    k = 10
    for embedding_way in tqdm(embedded_datasets):
        print()
        print()
        print()
        print("====================", embedding_way, "====================")
        for classifier in tqdm(classifiers):
            # X_trains, X_tests, y_trains, y_tests = large_embedding(embedding_way)
            X_trains, X_tests, y_trains, y_tests = embedding(embedding_way)
            for i in range(k):
                X_train = X_trains[i]
                y_train = y_trains[i]
                X_test = X_tests[i]
                y_test = y_tests[i]
                average_pre_score, ham_loss, one_error, acc_score, ranking_loss = \
                    globals()[classifier](X_train, X_test, y_train, y_test)
                # print(i + 1, "折评价指标：", average_pre_score, ham_loss, one_error, acc_score, ranking_loss)
                ap_result[classifier] = ap_result[classifier] + average_pre_score
                hamming_loss_result[classifier] = hamming_loss_result[classifier] + ham_loss
                one_error_result[classifier] = one_error_result[classifier] + one_error
                acc_score_result[classifier] = acc_score_result[classifier] + acc_score
                ranking_loss_result[classifier] = ranking_loss_result[classifier] + ranking_loss
            my_table.add_row([embedding_way, classifier, ap_result[classifier]/k,
                              hamming_loss_result[classifier]/k,
                              one_error_result[classifier]/k,
                              acc_score_result[classifier]/k,
                              ranking_loss_result[classifier]/k])
            # 将结果汇总为方便matlab制图的格式打印出来
            ap_m.append(ap_result[classifier]/k)
            hamming_loss_m.append(hamming_loss_result[classifier]/k)
            one_error_m.append(one_error_result[classifier]/k)
            acc_m.append(acc_score_result[classifier]/k)
            ranking_loss_m.append(ranking_loss_result[classifier]/k)
            for method in evaluates:
                method[classifier] = 0
        print(embedding_way, "for matlab result: ")
        print("ap=", ap_m, ";")
        print("hl=", hamming_loss_m, ";")
        print("or=", one_error_m, ";")
        print("acc= ", acc_m, ";")
        print("rl=", ranking_loss_m, ";")
        # print("ap_" + embedding_way[-8] + "_" + embedding_way[-5] + "=max(ap);")
        # print("hl_" + embedding_way[-8] + "_" + embedding_way[-5] + "=max(hl);")
        # print("or_" + embedding_way[-8] + "_" + embedding_way[-5] + "=max(or);")
        # print("acc_" + embedding_way[-8] + "_" + embedding_way[-5] + "=max(acc);")
        # print("rl_" + embedding_way[-8] + "_" + embedding_way[-5] + "=max(rl);")
        ap_m.clear()
        hamming_loss_m.clear()
        one_error_m.clear()
        acc_m.clear()
        ranking_loss_m.clear()
    print(my_table)
    # f = open("results/pse_param2.txt", "a+")
    # f.write(str(my_table))
    # f.close()

