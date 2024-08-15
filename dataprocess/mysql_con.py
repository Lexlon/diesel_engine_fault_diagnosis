import mysql.connector
import pandas as pd
from mysql.connector import Error

# 读取 CSV 文件
csv_file_path = 'data/minmax_1500.csv'
df = pd.read_csv(csv_file_path)

# 获取列名
columns = df.columns

# 生成创建表的 SQL 语句
create_table_query = '''
CREATE TABLE IF NOT EXISTS minmax_1500 (
    `AI1-02 [m/s²]` FLOAT(17,16),
    `AI1-05 [m/s²]` FLOAT(17,16),
    `AI1-08 [m/s²]` FLOAT(17,16),
    `AI1-11 [m/s²]` FLOAT(17,16),
    `AI1-14 [m/s²]` FLOAT(17,16)
)
'''

# 连接到 MySQL 数据库并创建表
try:
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='lxl19991029',  # 替换为你的 MySQL 密码
        database='fault_diagnosis'   # 替换为你的数据库名称
    )

    if connection.is_connected():
        cursor = connection.cursor()

        # 创建表格（如果尚未存在）
        cursor.execute(create_table_query)
        connection.commit()
        print("表格已创建或已存在")

except Error as e:
    print(f"Error: {e}")

finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed")

# 重新连接到 MySQL 数据库并插入数据
try:
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='lxl19991029',  # 替换为你的 MySQL 密码
        database='fault_diagnosis'   # 替换为你的数据库名称
    )

    if connection.is_connected():
        cursor = connection.cursor()

        # 将 DataFrame 中的数据插入到 MySQL 表中
        for i, row in df.iterrows():
            sql = '''
            INSERT INTO minmax_1500 (`AI1-02 [m/s²]`, `AI1-05 [m/s²]`, `AI1-08 [m/s²]`, `AI1-11 [m/s²]`, `AI1-14 [m/s²]`)
            VALUES (%s, %s, %s, %s, %s)
            '''
            cursor.execute(sql, tuple(row))

        connection.commit()
        print("数据已成功插入到数据库中")

except Error as e:
    print(f"Error: {e}")

finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed")