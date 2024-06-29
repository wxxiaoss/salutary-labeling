import os
import json
import pandas as pd
import numpy as np
from collections import Counter
from typing import Tuple, Mapping, Optional
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class DataTemplate():
    def __init__(self, x_train, y_train, s_train, x_val, y_val, s_val, x_test, y_test, s_test, l2_reg, s_col_idx):
        self.num_train: int = x_train.shape[0]
        self.num_val: int = x_val.shape[0]
        self.num_test: int = x_test.shape[0]
        self.dim: int = x_train.shape[1]
        self.num_s_feat = len(Counter(s_train))
        self.l2_reg = l2_reg
        self.s_col_idx = s_col_idx

        self.x_train: np.ndarray = x_train
        self.y_train: np.ndarray = y_train
        self.s_train: np.ndarray = s_train
        self.x_val: np.ndarray = x_val
        self.y_val: np.ndarray = y_val
        self.s_val: np.ndarray = s_val
        self.x_test: np.ndarray = x_test
        self.y_test: np.ndarray = y_test
        self.s_test: np.ndarray = s_test

        print("Dataset statistic - #total: %d; #train: %d; #val.: %d; #test: %d; #dim.: %.d\n"
              % (self.num_train + self.num_val + self.num_test,
                 self.num_train, self.num_val, self.num_test, self.dim))


class Dataset():
    """
    General dataset
    Assure in a binary group case, Grp. 1 is the privileged group and Grp. 0 is the unprivileged group
    Assure in a binary label case, 1. is the positive outcome and 0. is the negative outcome
    Sensitive feature is not excluded from data
    """

    def __init__(self, name, df, target_feat, sensitive_feat, l2_reg, test_df=None, categorical_feat=None,
                 drop_feat=None, s_thr=None, label_mapping=None, shuffle=False, load_idx=True, idx_path=None,
                 test_p=0.20, val_p=0.25, *args, **kwargs):
        """

        :param name: dataset name
        :param df: dataset DataFrame
        :param target_feat: feature to be predicted
        :param sensitive_feat: sensitive feature
        :param l2_reg: strength of l2 regularization for logistic regression model
        :param test_df: DataFrame for testing, optional
        :param categorical_feat: categorical features to be processed into one-hot encoding
        :param drop_feat: features to drop
        :param s_thr: threshold to split the data into two group, only for continuous sensitive feature
        :param label_mapping: mapping for one-hot encoding for some features
        :param shuffle: shuffle the dataset
        :param load_idx: loading shuffled row index
        :param idx_path: path for the shuffled index file
        :param test_p: proportion of test data
        :param val_p: proportion of validation data
        """

        print("Loading %s dataset.." % name)

        self.categorical_feat = categorical_feat if categorical_feat is not None else []
        self.l2_reg = l2_reg

        if shuffle:
            if load_idx and os.path.exists(idx_path):
                with open(idx_path) as f:
                    shuffle_idx = json.load(f)
            else:
                shuffle_idx = np.random.permutation(df.index)
                with open(idx_path, "w") as f:
                    json.dump(shuffle_idx.tolist(), f)

            df = df.reindex(shuffle_idx)

        df.dropna(inplace=True)
        if drop_feat is not None:
            df.drop(columns=drop_feat, inplace=True)

        if test_df is None:
            num_test = round(len(df) * test_p)
            num_train_val = len(df) - num_test
            train_val_df = df.iloc[:num_train_val]
            test_df = df.iloc[num_train_val:]
        else:
            test_df.dropna(inplace=True)
            if drop_feat is not None:
                test_df.drop(columns=drop_feat, inplace=True)
            train_val_df = df

        s_train_val, s_test = train_val_df[sensitive_feat].to_numpy(), test_df[sensitive_feat].to_numpy()
        if s_thr is not None:
            s_train_val = np.where(s_train_val >= s_thr[0], s_thr[1]["larger"], s_thr[1]["smaller"])
            s_test = np.where(s_test > s_thr[0], s_thr[1]["larger"], s_thr[1]["smaller"])
        else:
            assert sensitive_feat in label_mapping
            s_train_val = np.array([label_mapping[sensitive_feat][e] for e in s_train_val])
            s_test = np.array([label_mapping[sensitive_feat][e] for e in s_test])

        train_val_df, updated_label_mapping = self.one_hot(train_val_df, label_mapping)
        test_df, _ = self.one_hot(test_df, updated_label_mapping)

        y_train_val, y_test = train_val_df[target_feat].to_numpy(), test_df[target_feat].to_numpy()
        train_val_df, test_df = train_val_df.drop(columns=target_feat), test_df.drop(columns=target_feat)

        num_val = round(len(train_val_df) * val_p)
        num_train = len(train_val_df) - num_val
        x_train, x_val = train_val_df.iloc[:num_train], train_val_df.iloc[num_train:]
        self.y_train, self.y_val = y_train_val[:num_train], y_train_val[num_train:]
        self.s_train, self.s_val = s_train_val[:num_train], s_train_val[num_train:]
        self.y_test, self.s_test = y_test, s_test

        self.x_train, scaler = self.center(x_train)
        self.x_val, _ = self.center(x_val, scaler)
        self.x_test, _ = self.center(test_df, scaler)

        self.s_col_idx = train_val_df.columns.tolist().index(sensitive_feat)

        if name.startswith("DistShiftAdult") or name == "Adult" or name == "German" or name == "Synthetic" or name =='ACS' or name =='Credit' or name == 'Bank' or name=='CelebA' or name=='NLP':
            #Reconstituting training set by adding validation set back to training set 
            self.x_train = np.vstack((self.x_train, self.x_val))
            self.y_train = np.hstack((self.y_train, self.y_val))
            self.s_train = np.hstack((self.s_train, self.s_val))
            ##print(self.x_train.shape, self.y_train.shape, self.s_train.shape)
            #print(self.x_train.shape, self.y_train.shape)


            #Divide test set into validation and test set
            self.x_test, self.x_val, self.y_test, self.y_val, self.s_test, self.s_val = train_test_split(self.x_test, self.y_test, self.s_test, test_size=0.5, random_state=42000, shuffle=True)

            #np.save('temp-bins/xz_train.npy', self.x_train)
            #np.save('temp-bins/y_train.npy', self.y_train)
            #np.save('temp-bins/z_train.npy', self.s_train)
            #np.save('temp-bins/xz_test.npy', self.x_test)
            #np.save('temp-bins/y_test.npy', self.y_test)
            #np.save('temp-bins/z_test.npy', self.s_test)
            #np.save('temp-bins/xz_val.npy', self.x_val)
            #np.save('temp-bins/y_val.npy', self.y_val)
            #np.save('temp-bins/z_val.npy', self.s_val)
            

            #self.x_test, self.x_val = self.x_val, self.x_test
            #self.y_test, self.y_val = self.y_val, self.y_test
            #self.s_test, self.s_val = self.s_val, self.s_test

            #t = np.load('2trn.npy')
            #self.x_train, self.y_train = t[:,:101], t[:,101:].reshape((-1))
            #self.x_train, self.y_train = t[:,:102], t[:,102:].reshape((-1))
            #self.x_train, self.y_train = t[:,:56], t[:,56:].reshape((-1))

            #print(self.x_train.shape, self.y_train.shape)
            #print(updated_label_mapping)

            ##print(self.x_test.shape, self.x_val.shape)


        if name.startswith('RAA_Attack') or name.startswith('NRAA_Attack') or name.startswith('IAF_Attack') or name.startswith('Solans_Attack'):
            self.x_train = np.vstack((self.x_train, self.x_val))
            self.y_train = np.hstack((self.y_train, self.y_val))
            self.s_train = np.hstack((self.s_train, self.s_val))
            
            self.x_test, self.y_test, self.s_test, self.x_val, self.y_val, self.s_val  = self.x_test, self.y_test, self.s_test, self.x_test, self.y_test, self.s_test


        if name == "Toy":
            #Removing sensitive attribute for training
            self.x_train, self.x_val, self.x_test = self.x_train[:,:-1], self.x_val[:,:-1], self.x_test[:,:-1]

            #Reconstituting training set by adding validation set back to training set 
            self.x_train = np.vstack((self.x_train, self.x_val))
            self.y_train = np.hstack((self.y_train, self.y_val))
            self.s_train = np.hstack((self.s_train, self.s_val))
            ##print(self.x_train.shape, self.y_train.shape, self.s_train.shape)

            #Divide test set into validation and test set
            ##self.x_test, self.x_val, self.y_test, self.y_val, self.s_test, self.s_val = train_test_split(self.x_test, self.y_test, self.s_test, test_size=0.6, random_state=42, shuffle=False)
            self.x_val = self.x_test
            self.y_val = self.y_test
            self.s_val = self.s_test

            #self.x_test, self.x_val = self.x_val, self.x_test
            #self.y_test, self.y_val = self.y_val, self.y_test
            #self.s_test, self.s_val = self.s_val, self.s_test

            ##print(self.x_test.shape, self.x_val.shape)



    def one_hot(self, df: pd.DataFrame, label_mapping: Optional[Mapping]) -> Tuple[pd.DataFrame, Mapping]:
        label_mapping = {} if label_mapping is None else label_mapping
        updated_label_mapping = {}
        for c in df.columns:
            if c in self.categorical_feat:
                column = df[c]
                df = df.drop(c, axis=1)

                if c in label_mapping:
                    mapping = label_mapping[c]
                else:
                    unique_values = list(dict.fromkeys(column))
                    mapping = {v: i for i, v in enumerate(unique_values)}
                    updated_label_mapping[c] = mapping

                n = len(mapping)
                if n > 2:
                    for i in range(n):
                        col_name = '{}.{}'.format(c, i)
                        col_i = [1. if list(mapping.keys())[i] == e else 0. for e in column]
                        df[col_name] = col_i
                else:
                    col = [mapping[e] for e in column]
                    df[c] = col

        updated_label_mapping.update(label_mapping)

        return df, updated_label_mapping

    @staticmethod
    def center(X: pd.DataFrame, scaler: preprocessing.StandardScaler = None) -> Tuple:
        if scaler is None:
            scaler = preprocessing.StandardScaler().fit(X.values)
        scaled = scaler.transform(X.values)

        return scaled, scaler

    @property
    def data(self):
        return DataTemplate(self.x_train, self.y_train, self.s_train,
                            self.x_val, self.y_val, self.s_val,
                            self.x_test, self.y_test, self.s_test,
                            self.l2_reg, self.s_col_idx)




class Bank(Dataset):

    def __init__(self):
        meta = json.load(open("./data/bank/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        #column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], skipinitialspace=True)
        test = pd.read_csv(meta["test_path"],  skipinitialspace=True)

        
        super(Bank, self).__init__(name="Bank", df=train, test_df=test, **meta, shuffle=False)





class Toy(Dataset):

    def __init__(self):
        meta = json.load(open("./data/toy/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], names=column_names, skipinitialspace=True,
                            na_values=meta["na_values"])
        test = pd.read_csv(meta["test_path"], names=column_names, skipinitialspace=True,
                           na_values=meta["na_values"])        


        super(Toy, self).__init__(name="Toy", df=train, test_df=test, **meta, shuffle=False)





class CompasDataset(Dataset):
    """ https://github.com/propublica/compas-analysis """

    def __init__(self):
        meta = json.load(open("./data/compas/meta.json"))

        df = pd.read_csv(meta["train_path"], index_col='id')
        df = self.default_preprocessing(df)
        df = df[meta["features_to_keep"].split(",")]

        super(CompasDataset, self).__init__(name="Compas", df=df, **meta, shuffle=False)

    @staticmethod
    def default_preprocessing(df):
        """
        Perform the same preprocessing as the original analysis:
        https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
        """

        def race(row):
            return 'Caucasian' if row['race'] == "Caucasian" else 'Not Caucasian'

        def two_year_recid(row):
            return 'Did recid.' if row['two_year_recid'] == 1 else 'No recid.'

        df['race'] = df.apply(lambda row: race(row), axis=1)
        df['two_year_recid'] = df.apply(lambda row: two_year_recid(row), axis=1)

        return df[(df.days_b_screening_arrest <= 30)
                  & (df.days_b_screening_arrest >= -30)
                  & (df.is_recid != -1)
                  & (df.c_charge_degree != 'O')
                  & (df.score_text != 'N/A')]


def fair_stat(data: DataTemplate):
    s_cnt = Counter(data.s_train)
    s_pos_cnt = {s: 0. for s in s_cnt.keys()}
    for i in range(data.num_train):
        if data.y_train[i] == 1:
            s_pos_cnt[data.s_train[i]] += 1

    print("-" * 10, "Statistic of fairness")
    for s in s_cnt.keys():
        print("Grp. %d - #instance: %d; #pos.: %d; ratio: %.3f" % (s, s_cnt[s], s_pos_cnt[s], s_pos_cnt[s] / s_cnt[s]))

    print("Overall - #instance: %d; #pos.: %d; ratio: %.3f" % (sum(s_cnt.values()), sum(s_pos_cnt.values()),
                                                               sum(s_pos_cnt.values()) / sum(s_cnt.values())))

    return


def fetch_data(name):
    if  name == "compas":
        return CompasDataset().data
    elif name == "bank":
        return Bank().data
    elif name == "toy":
        return Toy().data
    else:
        raise ValueError


if __name__ == "__main__":
    data = fetch_data("bank")
    fair_stat(data)
