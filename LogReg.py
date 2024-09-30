from fileinput import filename

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 0. Создание класса для линейной регрессии
class LogReg:
    def __init__(self, learning_rate, n_inputs, n_epochs=1000):
        self.learning_rate = learning_rate
        self.n_inputs = n_inputs
        self.n_epochs = n_epochs
        self.coef_ = np.random.uniform(0, 1, n_inputs)
        self.intercept_ = np.random.uniform(0, 1, 1)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        def sigmoid(x):
            sigmoid = 1 / (1 + np.exp(-x))
            return sigmoid

        for epoch in range(1, self.n_epochs + 1):
            y_pred = sigmoid(np.dot(X, self.coef_) + self.intercept_)

            dw = (-1 / self.n_inputs) * np.dot(X.T, (y - y_pred))
            dw0 = (-1 / self.n_inputs) * np.sum((y - y_pred))

            self.coef_ = self.coef_ - self.learning_rate * dw
            self.intercept_ = self.intercept_ - self.learning_rate * dw0

        return self.coef_, self.intercept_

    def predict(self, X):
        def sigmoid(x):
            sigmoid = 1 / (1 + np.exp(-x))
            return sigmoid

        y_pred = np.round(sigmoid(np.dot(X, self.coef_) + self.intercept_))
        return pd.DataFrame(y_pred, columns = ['Prediction'])

# 1. Заголовок
st.title("Machine Learning by Logistic Regression")
st.write("Загрузи Data Frame в формате .csv в окошке слева, обучи свою модель и получи предсказание!")

# 2. Загрузка дата фрейма и выбор таргета и фичей
uploaded_file = st.sidebar.file_uploader('Загрузи CSV файл', type='csv')
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader('Загруженный Data Frame:')
        st.write(df)
        target = st.sidebar.selectbox("Выбери название колонки для target:", df.columns)
        #target = st.sidebar.text_input("Введи название колонки для target:")

        # Cоздание списка фичей и таргета
        features = df.drop(target, axis=1).columns

        # Нормировка
        ss_scaler = StandardScaler()
        df[features] = ss_scaler.fit_transform(df[features])

 # 4. Ввод пользователем learning rate и создание экземпляра класса
        lr = st.sidebar.number_input("Введи значение learning rate (от 0.0001 до 0.1)", value=0.01, step=0.0001)

        my_logreg = LogReg(learning_rate=lr, n_inputs=(df.shape[1] - 1))
        my_logreg.fit(df[features], df[target])

# 5. Вывод результатов логистической регрессии
        values = [w for w in my_logreg.coef_]
        res_dict = dict(zip(features, values))
        res_dict['w0'] = my_logreg.intercept_[0]

        st.subheader('Результаты регрессии - подобранные веса для каждого признака:')
        st.write(res_dict)

 # 6. Построение скаттер плот
        st.sidebar.subheader('Выбери два признака, по которым ты хочешь построить Scatter Plot')

        scatter_feat1 = st.sidebar.selectbox("Введи название первого признака:", df.columns)
        scatter_feat2 = st.sidebar.selectbox("Введи название второго признака:", df.columns)

        st.subheader(f'Scatter Plot по признакам: {scatter_feat1} и {scatter_feat2} c разделяющей прямой от логистической регрессии')

        colors = df[target].map({1: 'red', 0: 'blue'})
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(df[scatter_feat1], df[scatter_feat2], c=colors, alpha=0.7)
        plt.xlabel(scatter_feat1)
        plt.ylabel(scatter_feat2)

        plt.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', label=f'{target} = 1',
                            markersize=10, markerfacecolor='red'),
            plt.Line2D([0], [0], marker='o', color='w', label=f'{target} = 0',
                               markersize=10, markerfacecolor='blue')
        ])
        line_x = np.linspace(df[scatter_feat1].min(), df[scatter_feat1].max(), 1000)
        line_y = (-my_logreg.coef_[0] * line_x - my_logreg.intercept_) / my_logreg.coef_[1]
        plt.plot(line_x, line_y, c='black')
        plt.grid()
        st.pyplot(fig)

# 6. Получение предсказаний
        uploaded_file_1 = st.sidebar.file_uploader('Загрузи CSV файл, по которому хочешь получить предсказание', type='csv')
        if uploaded_file_1 is not None:
            try:
                df1 = pd.read_csv(uploaded_file_1)
                st.subheader('Загруженный Data Frame для предсказания:')
                st.write(df1)

                st.subheader(f'Предсказания {target}:')
                df1[features] = ss_scaler.transform(df1[features])
                st.write(my_logreg.predict(df1[features]))

            except Exception as e:
                st.error(f"Не удалось открыть Data Frame. Ошибка: {e}")


    except Exception as e:
        st.error(f"Не удалось открыть Data Frame. Ошибка: {e}")