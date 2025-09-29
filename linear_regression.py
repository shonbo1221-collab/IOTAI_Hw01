
import matplotlib
matplotlib.use('Agg') # Must be called before import matplotlib.pyplot as plt

from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

app = Flask(__name__)

# 設定中文字體
font = fm.FontProperties(fname=r'c:\windows\fonts\msjh.ttc')

@app.route('/', methods=['GET', 'POST'])
def index():
    # 預設值
    default_a = np.random.uniform(-10, 10)
    default_b = np.random.uniform(-10, 10) # b 係數固定，不從表單獲取
    default_num_points = np.random.randint(100, 1001)
    default_noise_variance = np.random.uniform(0, 1000)

    if request.method == 'POST':
        a = float(request.form.get('a', default_a))
        # b 係數不從表單獲取，使用預設值
        b = default_b
        num_points = int(request.form.get('num_points', default_num_points))
        noise_variance = float(request.form.get('noise_variance', default_noise_variance))
    else:
        a = default_a
        b = default_b
        num_points = default_num_points
        noise_variance = default_noise_variance

    # 產生隨機的 x 資料
    x = np.random.rand(num_points, 1) * 100

    # 根據 y = ax + b 產生 y 資料，並加上一些雜訊
    noise = np.random.randn(num_points, 1) * noise_variance
    y = a * x + b + noise

    # 建立並訓練線性回歸模型 (含截距)
    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)

    # 取得預測的 a 和 b
    a_pred = model.coef_[0][0]
    b_pred = model.intercept_[0]

    # 計算每個點到回歸線的距離
    y_pred = model.predict(x)
    distances = np.abs(y - y_pred)

    # 找出距離最遠的 5 個點的索引
    # 確保 num_points 至少為 5，否則取 num_points 個點
    num_outliers = min(5, num_points)
    furthest_indices = np.argsort(distances.flatten())[-num_outliers:]

    # 視覺化結果
    plt.figure()
    plt.clf() # 清除當前圖形
    plt.cla() # 清除當前軸

    # 繪製所有點
    plt.scatter(x, y, label='原始資料', color='blue')

    # 繪製最遠的 5 個點
    plt.scatter(x[furthest_indices], y[furthest_indices], color='green', s=100, label=f'最遠的 {num_outliers} 個點', edgecolors='black', zorder=5)

    plt.plot(x, model.predict(x), color='red', linewidth=3, label='回歸線')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('線性回歸結果 (y = ax + b)', fontproperties=font)
    plt.legend(prop=font)
    
    # 儲存圖片
    if not os.path.exists('static'):
        os.makedirs('static')
    plot_path = os.path.join('static', 'plot.png')
    plt.savefig(plot_path)
    plt.close()

    return render_template('index.html', 
                           num_points=num_points,
                           original_a=f'{a:.4f}',
                           original_b=f'{b:.4f}',
                           predicted_a=f'{a_pred:.4f}',
                           predicted_b=f'{b_pred:.4f}',
                           noise_variance=f'{noise_variance:.4f}')

if __name__ == '__main__':
    app.run(debug=False)
