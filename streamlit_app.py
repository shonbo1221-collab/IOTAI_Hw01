
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 設定中文字體
# 檢查系統中是否存在微軟正黑體，如果不存在則使用預設字體
try:
    font_path = fm.findfont(fm.FontProperties(family='Microsoft JhengHei'))
    font_properties = fm.FontProperties(fname=font_path)
except:
    st.warning("找不到 'Microsoft JhengHei' 字體，將使用 Matplotlib 預設字體。")
    font_properties = fm.FontProperties()

plt.rcParams['font.family'] = font_properties.get_name()
plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題

st.set_page_config(layout="wide", page_title="線性回歸結果")

st.title("線性回歸結果")

# 側邊欄參數設定
st.sidebar.header("參數設定")

default_a = np.random.uniform(-10, 10)
default_b = np.random.uniform(-10, 10)
default_num_points = np.random.randint(100, 1001)
default_noise_variance = np.random.uniform(0, 1000)

a = st.sidebar.slider(
    "係數 a",
    min_value=-10.0,
    max_value=10.0,
    value=float(st.session_state.get('a', default_a)),
    step=0.1,
    key='a_slider'
)
num_points = st.sidebar.slider(
    "點的數量",
    min_value=100,
    max_value=1000,
    value=int(st.session_state.get('num_points', default_num_points)),
    step=10,
    key='num_points_slider'
)
noise_variance = st.sidebar.slider(
    "雜訊方差",
    min_value=0.0,
    max_value=1000.0,
    value=float(st.session_state.get('noise_variance', default_noise_variance)),
    step=1.0,
    key='noise_variance_slider'
)

# 將當前值存儲到 session_state，以便重新運行時保持
st.session_state['a'] = a
st.session_state['num_points'] = num_points
st.session_state['noise_variance'] = noise_variance

# 產生隨機的 x 資料
x = np.random.rand(num_points, 1) * 100

# 根據 y = ax + b 產生 y 資料，並加上一些雜訊
# b 係數不從表單獲取，使用預設值 (這裡我們讓它每次運行都隨機生成，或者可以固定)
b = default_b # 這裡使用預設的 b，保持與原 Flask 應用一致
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
num_outliers = min(5, num_points)
furthest_indices = np.argsort(distances.flatten())[-num_outliers:]

# 視覺化結果
fig, ax = plt.subplots(figsize=(10, 6))

# 繪製所有點
ax.scatter(x, y, label='原始資料', color='blue')

# 繪製最遠的 5 個點
ax.scatter(x[furthest_indices], y[furthest_indices], color='green', s=100, label=f'最遠的 {num_outliers} 個點', edgecolors='black', zorder=5)

ax.plot(x, model.predict(x), color='red', linewidth=3, label='回歸線')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('線性回歸結果 (y = ax + b)')
ax.legend()

st.pyplot(fig)

st.subheader("結果摘要")
st.write(f"隨機產生 **{num_points}** 個點")
st.write(f"原始 a: **{a:.4f}**")
st.write(f"原始 b: **{b:.4f}**")
st.write(f"預測 a: **{a_pred:.4f}**")
st.write(f"預測 b: **{b_pred:.4f}**")
st.write(f"雜訊方差: **{noise_variance:.4f}**")

# 為了與原 Flask 應用程式的重新整理按鈕行為一致，Streamlit 會在參數改變時自動重新執行
# 所以不需要額外的重新整理按鈕
