import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.preprocessing import StandardScaler

class LogReg:
    def __init__(self, learning_rate, n_inputs):
        self.learning_rate = learning_rate
        self.n_inputs = n_inputs
        self.coef_ = np.random.uniform(-1, 1, n_inputs)
        self.intercept_ = np.random.uniform(-1, 1)
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def fit(self, X, y, n_epochs):
        for epoch in range(n_epochs):
            y_pred = self.predict(X)
            error = y_pred - y
            gradient_coef = np.dot(X.T, error) / len(y)
            gradient_intercept = np.sum(error) / len(y)
            
            self.coef_ -= self.learning_rate * gradient_coef
            self.intercept_ -= self.learning_rate * gradient_intercept

    def predict(self, X):
        linear_model_z = self.intercept_ + np.dot(X, self.coef_)
        y_pred = self.sigmoid(linear_model_z)
        return y_pred
    
    def score(self, X, y):
        mean_y_true = np.mean(y)
 
        ss_res = np.sum((y - self.predict(X)) ** 2)
        ss_tot = np.sum((y - mean_y_true) ** 2)
 
        r2 = 1 - (ss_res / ss_tot)
        
        return r2
    
    def dec_plot(self, X, y):
        plt.figure(figsize=(12, 6))
        colors = ['blue' if loan == 0 else 'red' for loan in y]
        plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.5)

        x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 1000)
        y_values = - self.coef_[0]/self.coef_[1] * x_values - self.intercept_/self.coef_[1]
        plt.plot(x_values, y_values, label='Decision Boundary', color='black')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        # plt.show()
        
        st.pyplot(plt)
        

st.title('Logistic regression')
# st.header('Example')

# train = pd.read_csv('~/Desktop/ds_bootcamp/Phase_1/ds-phase-1/05-math/aux/credit_train.csv')
# test = pd.read_csv('~/Desktop/ds_bootcamp/Phase_1/ds-phase-1/05-math/aux/credit_test.csv')

scaler = StandardScaler()

# X_train = train[['CCAvg', 'Income']]
# y_train = train['Personal.Loan']
# X_test = test[['CCAvg', 'Income']]
# y_test = test['Personal.Loan']

# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# learning_rate = st.write('Learning rate num: 0.01')
# n_epochs = st.write('Input number of Epochs: 10000')

# model = LogReg(learning_rate=0.01, n_inputs=X_train.shape[1])
# model.fit(X_train_scaled, y_train, n_epochs=10000)

# #Proveryaem na testovom
# predictions_for_test = model.predict(X_test_scaled)
# #Output
# st.write(f"Model coefficients: w1: {model.coef_[0]}, w2: {model.coef_[1]}")
# st.write("Model intercept:", model.intercept_)
# st.write("Model accuracy on test set:", model.score(X_test_scaled, y_test))

# model.dec_plot(X_test_scaled, y_test)

#These part of code for own data set
st.subheader('Own data set')
data = st.file_uploader("Upload csv file", type="csv")
if data:
    df = pd.read_csv(data)
    
    st.write(", ".join(df.columns))
    target = st.text_input("Enter target column name")
    if target in df:
        y = df[target]
        st.write("Whcih features you want to include to analyze")
        features = st.text_input("Enter names of the features, separated by commas(,)").split(',')
        features = [feature.strip() for feature in features if feature.strip() in df.columns]
        
        if features:
            X = df[features]
        
            lr = st.number_input('Input learning rate number:', value=0.01, step=0.005)
            epochs = st.number_input('Input number of Epochs:', value=100, step=20)
        
            X_scaled = scaler.fit_transform(X)
            y = y.values
        
            own_model = LogReg(learning_rate=lr, n_inputs=X.shape[1])
            own_model.fit(X_scaled, y, n_epochs=epochs)
        
            prediction = own_model.predict(X_scaled)
            st.write(f"Last prediction: {prediction[-1]}")
            
            coef_dict = {features[i]: own_model.coef_[i] for i in range(len(features))}
            st.write("Feature weights:", coef_dict)
            st.write("Accuracy:", {own_model.score(X_scaled, y)})
        
            st.write('If you want to see the graph of two features, type the name of the columns:')
            col_name_for_graph = st.text_input('Enter names, separated by commas(,):').split(',')
            col_name_for_graph = [col.strip() for col in col_name_for_graph if col.strip() in features]
            
            if len(col_name_for_graph) == 2:
                features_indices = [features.index(col) for col in col_name_for_graph]
                X_for_plot = X_scaled[:, features_indices]
                own_model.dec_plot(X_for_plot, y)
            
            else:
                st.write("Enter valid names")
        else:
            st.write("Enter valid features names")
        
else:
    st.stop()
