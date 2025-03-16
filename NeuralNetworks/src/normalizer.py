from sklearn.preprocessing import MinMaxScaler

class Normalizer:
    def __init__(self):
        self.scaler_X = MinMaxScaler()
        self.scaler_Y = MinMaxScaler()

    def fit_transform(self, X_train, Y_train):
        X_train_normalized = self.scaler_X.fit_transform(X_train.reshape(-1, 1))
        Y_train_normalized = self.scaler_Y.fit_transform(Y_train.reshape(-1, 1))
        return X_train_normalized, Y_train_normalized

    def transform(self, X_test, Y_test):
        X_test_normalized = self.scaler_X.transform(X_test.reshape(-1, 1))
        Y_test_normalized = self.scaler_Y.transform(Y_test.reshape(-1, 1))
        return X_test_normalized, Y_test_normalized

    def denormalize_X(self, X_normalized):
        return self.scaler_X.inverse_transform(X_normalized)

    def denormalize_Y(self, Y_normalized):
        return self.scaler_Y.inverse_transform(Y_normalized)
