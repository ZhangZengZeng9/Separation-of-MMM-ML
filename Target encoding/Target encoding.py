import pandas as pd
import random
import numpy as np
from sklearn import base
from sklearn.model_selection import KFold

read_file1 = pd.read_excel("permeability.xlsx")
read_file1.to_csv("permeability.csv",
                 index = None,
                 header = True)
df1 = pd.DataFrame(pd.read_csv("permeability.csv"))

read_file2 = pd.read_excel("permeability-t.xlsx")
read_file2.to_csv("permeability-t.csv",
                 index = None,
                 header = True)
df2 = pd.DataFrame(pd.read_csv("permeability-t.csv"))


def getRandomDataFrame(data, numCol):
    if data == 'permeability':
        fillers = random.choices(['PSF ', 'ODPA-TMPDA', 'PDMS', 'Trögers base polymer',
                                  '6FDA-ODA', 'PES', 'Pebax 1657', ' 1185 A-10',
                                  'PVI-POEM', 'SBS-g-POEM', 'PEBA 2533', '6FDA-BI',
                                  'Pebax 2533', 'Pebax MH 1657', ' Matrimid  5218', 'PEBA',
                                  'Pebax', 'PEG/PPG-PDMS', 'PEGDA', 'PMP'], k=numCol)
        value = np.random.randint(2, size=(numCol,))
        df = pd.DataFrame({'Fillers': fillers, 'Selectivity(CO2/N2)': value})
        return df
    elif data == 'permeability-t':
        fillers = random.choices(['PSF ', 'ODPA-TMPDA', 'PDMS', 'Trögers base polymer',
                                  '6FDA-ODA', 'PES', 'Pebax 1657', ' 1185 A-10',
                                  'PVI-POEM', 'SBS-g-POEM', 'PEBA 2533', '6FDA-BI',
                                  'Pebax 2533', 'Pebax MH 1657', ' Matrimid  5218', 'PEBA',
                                  'Pebax', 'PEG/PPG-PDMS', 'PEGDA', 'PMP'], k=numCol)
        df = pd.DataFrame({'Fillers': fillers})
        return df
    else:
        print(';)')


train = pd.read_csv('./permeability.csv')
test = pd.read_csv('./permeability-t.csv')
print(train)
train.groupby('Fillers').mean()
print(test)

class KFoldTargetEncoderTrain(base.BaseEstimator, base.TransformerMixin):

        def __init__(self, colnames, targetName, n_fold=5, verbosity=True, discardOriginal_col=False):

            self.colnames = colnames
            self.targetName = targetName
            self.n_fold = n_fold
            self.verbosity = verbosity
            self.discardOriginal_col = discardOriginal_col

        def fit(self, X, y=None):
            return self

        def transform(self, X):

            assert (type(self.targetName) == str)
            assert (type(self.colnames) == str)
            assert (self.colnames in X.columns)
            assert (self.targetName in X.columns)

            mean_of_target = X[self.targetName].mean()
            kf = KFold(n_splits=self.n_fold, shuffle=True, random_state=2019)

            col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
            X[col_mean_name] = np.nan

            for tr_ind, val_ind in kf.split(X):
                X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]

                X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(
                    X_tr.groupby(self.colnames)[self.targetName].mean())

            X[col_mean_name].fillna(mean_of_target, inplace=True)

            if self.verbosity:
                encoded_feature = X[col_mean_name].values
                print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,
                                                                                      self.targetName,
                                                                                      np.corrcoef(
                                                                                          X[self.targetName].values,
                                                                                          encoded_feature)[0][1]))
            if self.discardOriginal_col:
                X = X.drop(self.targetName, axis=1)

            return X

targetc = KFoldTargetEncoderTrain('Fillers', 'Selectivity(CO2/N2)', n_fold=5)
new_train = targetc.fit_transform(train)
print(new_train)

df = getRandomDataFrame('permeability', 309)
result1 = df[['Fillers','Selectivity(CO2/N2)']].iloc[1:309,:].groupby('Fillers').mean()
result2 = df[['Fillers','Selectivity(CO2/N2)']].groupby('Fillers').mean()

result1.to_excel('result1.xlsx')
result2.to_excel('result2.xlsx')

# Display result
print("Result 1:")
print(result1)
print("\nResult 2:")
print(result2)

# Test
class KFoldTargetEncoderTest(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, train, colNames, encodedName):
        self.train = train
        self.colNames = colNames
        self.encodedName = encodedName

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mean = self.train[[self.colNames, self.encodedName]].groupby(self.colNames).mean().reset_index()

        dd = {}
        for index, row in mean.iterrows():
            dd[row[self.colNames]] = row[self.encodedName]

        X[self.encodedName] = X[self.colNames]
        X = X.replace({self.encodedName: dd})

        return X

test_targetc = KFoldTargetEncoderTest(new_train,'Fillers','Fillers_Kfold_Target_Enc')
test_targetc.fit_transform(test)
print(test_targetc.fit_transform(test))
result = test_targetc.fit_transform(test)
result_df = pd.DataFrame(result)
result_df.to_excel('result-T.xlsx')