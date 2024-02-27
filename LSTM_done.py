import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
import datetime
import os
import matplotlib.pyplot as plt
import openpyxl


def preprocess_data(data):
    # 归一
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


# 数据集
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


# 加载
def load_excel_data(filename):
    df = pd.read_excel(filename)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df


# 训练
def train_model(data, look_back):
    X, Y = create_dataset(data, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(4, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=3000, batch_size=8, verbose=2)

    return model



# 预测
def predict(model, scaler, data, look_back):
    data = np.reshape(data, (1, look_back, 1))
    data = scaler.transform(np.array(data).reshape(-1, 1))
    prediction = model.predict(data)
    prediction = scaler.inverse_transform(prediction)
    return prediction


# 主函数
if __name__ == "__main__":

    filename = "test.xlsx"  # 数据集
    model_path = "lstm_model.keras"  # 模型
    look_back = 1000

    # 加载数据
    df = load_excel_data(filename)
    data, scaler = preprocess_data(df['Total'].values.reshape(-1, 1))

    if not os.path.exists(model_path):
        print("开始训练模型...")
        model = train_model(data, look_back)  # 训练模型
        model.save(model_path)

    else:
        # 加载模型
        print("直接加载模型...")
        model = load_model(model_path)
        # 使用模型进行预测

    input_date = df.index[-1] - datetime.timedelta(days=look_back)
    input_sequence = df.loc[input_date:input_date + datetime.timedelta(days=look_back - 1), 'Total'].values
    pretime_sequence = []
    predictions = []
    for i in range(look_back + 365):
        # print(f"当前序列：{input_sequence}")
        prediction = predict(model, scaler, [input_sequence], look_back)

        pretime_sequence.append(pd.to_datetime(input_date))
        predictions.append(prediction[0][0])

        input_date += datetime.timedelta(days=1)
        input_sequence[:-1] = input_sequence[1:]
        input_sequence[-1] = prediction[0][0]

        print(f"{input_date}: 预测值: {str(prediction[0][0])}")
    # df.loc[input_date:, 'pre'] = predictions[look_back:]
    plt.plot(pretime_sequence, predictions, label='pre')

    # 绘制实际值
    plt.plot(df.index.values, df['Total'].values,
             label='ori')

    plt.xlabel('Time')
    plt.xlabel('Time')
    plt.ylabel('House Price')
    plt.legend()
    plt.show()

'''
plt.plot(pretime_sequence, predictions, label='Actual Values (Training Data)')

# 绘制实际值
plt.plot(df["date"], df['Total'].values,
         label='Actual Values (Training Data)')

plt.xlabel('Time')
plt.xlabel('Time')
plt.ylabel('House Price')
plt.legend()
plt.show()
'''