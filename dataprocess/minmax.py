from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def read_data(speed,label = 1):
    labeled_data = []
    data = pd.read_excel('../data/'+speed+'.xlsx')
    scaler = MinMaxScaler()
    columns_to_scale = ['AI1-02 [m/s²]', 'AI1-05 [m/s²]',
                        'AI1-08 [m/s²]', 'AI1-11 [m/s²]',
                        'AI1-14 [m/s²]']
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    return data[columns_to_scale]

if __name__ == '__main__':
    data_list = ['1500', '1800', '2100', '2300']
    for data in data_list:
        re = read_data(data)
        re.to_csv('../data/minmax_'+data+'.csv', index=False)

