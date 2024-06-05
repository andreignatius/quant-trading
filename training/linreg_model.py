from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel


class PCARSIModel(BaseModel):
    def __init__(self, file_path, train_start, train_end, test_start, test_end, trading_instrument):
        super().__init__(file_path, train_start, train_end, test_start, test_end, trading_instrument)
        
        self.rsi_lbs = [i for i in range(2,25)]
        self.n_components = 3
        self.lookahead = 6
        self.pca = PCA(n_components=self.n_components)
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.long_thresh = None
        self.short_thresh = None

    def train(self):
        self.train_test_split_time_series()

        print("y_train value counts: ", self.y_train.value_counts())
        # self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        # self.X_test_scaled = self.scaler.transform(self.X_test)

        # self.model.fit(self.X_train_scaled, self.y_train)
        print("check X train: ", self.X_train)
        print("check y train: ", self.y_train)
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        # self.data['PredictedLabel'] = self.model.predict(self.scaler.transform(self.X))
        # return self.data
        # predicted_categories = self.model.predict(self.scaler.transform(self.X_test_scaled))
        predicted_categories = self.model.predict(self.X_test)
        # print("CHECK predicted_labels: ", predicted_categories)
        return predicted_categories

    def evaluate(self, X, y):
        # Implement evaluation logic
        pass





class PCARSIModel:
    def __init__(self, rsi_lbs: List[int], n_components: int = 2, lookahead: int = 6):
        self.rsi_lbs = rsi_lbs
        self.n_components = n_components
        self.lookahead = lookahead
        self.pca = PCA(n_components=n_components)
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.long_thresh = None
        self.short_thresh = None
    
    def fit(self, ohlc: pd.DataFrame, train_size: int):
        rsis = pd.DataFrame()
        for lb in self.rsi_lbs:
            rsis[f'RSI_{lb}'] = ta.rsi(ohlc['close'], lb)

        warm_up = max(self.rsi_lbs) * 2
        train_data = rsis.iloc[warm_up:warm_up + train_size].dropna()
        y = np.log(ohlc['close']).diff(self.lookahead).shift(-self.lookahead).iloc[warm_up:warm_up + train_size]
        y = y.loc[train_data.index].dropna()
        train_data = train_data.loc[y.index]

        scaled_data = self.scaler.fit_transform(train_data)
        pca_data = self.pca.fit_transform(scaled_data)
        
        self.model.fit(pca_data, y)

        preds = self.model.predict(pca_data)
        self.long_thresh = np.quantile(preds, 0.99)
        self.short_thresh = np.quantile(preds, 0.01)
    
    def predict(self, ohlc: pd.DataFrame):
        rsis = pd.DataFrame()
        for lb in self.rsi_lbs:
            rsis[f'RSI_{lb}'] = ta.rsi(ohlc['close'], lb)

        scaled_data = self.scaler.transform(rsis)
        pca_data = self.pca.transform(scaled_data)
        preds = self.model.predict(pca_data)

        signals = np.where(preds > self.long_thresh, 1, np.where(preds < self.short_thresh, -1, 0))
        
        output_df = pd.DataFrame(index=ohlc.index)
        output_df['pred'] = preds
        output_df['long_thresh'] = self.long_thresh
        output_df['short_thresh'] = self.short_thresh
        output_df['signal'] = signals

        return output_df

# Example usage
ohlc = pd.DataFrame()  # Replace with your actual OHLC data
model = PCARSIModel(rsi_lbs=[14, 28], n_components=2, lookahead=6)
model.fit(ohlc, train_size=500)  # Specify your actual train size
predictions = model.predict(ohlc)
print(predictions)
