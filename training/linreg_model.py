from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel
import numpy as np
import pandas as pd

class PCARSIModel(BaseModel):
    def __init__(self, file_path, train_start, train_end, test_start, test_end, trading_instrument, model_name):
        super().__init__(file_path, train_start, train_end, test_start, test_end, trading_instrument, model_name)
        
        # self.rsi_lbs = [i for i in range(2,25)]
        self.n_components = 3
        self.lookahead = 6
        self.pca = PCA(n_components=self.n_components)
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.long_thresh = None
        self.short_thresh = None

    def fit(self):
        self.train_test_split_time_series()

        print("y_train value counts: ", self.y_train.value_counts())
        # self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        # self.X_test_scaled = self.scaler.transform(self.X_test)

        # self.model.fit(self.X_train_scaled, self.y_train)
        print("check X train: ", self.X_train)
        print("check y train: ", self.y_train)

        # pca_X_train = self.X_train.dropna()
        # pca_y_train = self.y_train.loc[pca_X_train.index].dropna()

        # Drop rows with NaN values in X_train
        pca_X_train = self.X_train.dropna()

        # Ensure y_train is aligned with the remaining X_train indices
        pca_y_train = self.y_train.loc[pca_X_train.index].dropna()

        # Align indices again in case any NaNs were dropped from y_train
        pca_X_train = pca_X_train.loc[pca_y_train.index]

        scaled_data = self.scaler.fit_transform(pca_X_train)
        pca_data = self.pca.fit_transform(scaled_data)
        
        self.model.fit(pca_data, pca_y_train)

        preds = self.model.predict(pca_data)
        self.long_thresh = np.quantile(preds, 0.99)
        self.short_thresh = np.quantile(preds, 0.01)

    def predict(self):
   
        rsis = self.X_test
        print("++++check rsis: ++++", rsis)
        scaled_data = self.scaler.transform(rsis)
        pca_data = self.pca.transform(scaled_data)
        preds = self.model.predict(pca_data)

        signals = np.where(preds > self.long_thresh, 1, np.where(preds < self.short_thresh, -1, 0))
        #Ensure that signals hold for 6 periods. If there is a same direction signal immediately after, won't enter new position but hold it for longer (e.g. count 6 days from new signal)
        rolling_signals = pd.Series(signals).rolling(window=self.lookahead).mean() 
        rolling_signals = np.where(abs(rolling_signals) > 0, np.sign(rolling_signals), 0)
        print("+++rolling_signals+++: ", rolling_signals)
        return rolling_signals
    ######## Fix for 6-period holding #################

    def evaluate(self, X, y):
        # Implement evaluation logic
        pass

