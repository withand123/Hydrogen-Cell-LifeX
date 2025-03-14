import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import webbrowser
import statsmodels.api as sm

# 设置页面标题和图标
st.set_page_config(page_title="Hydrogen Cell LifeX",
                   page_icon="./photo/顶部栏.png",
                   # layout="wide",
                   initial_sidebar_state="expanded"
                   )
# 显示 Logo
st.sidebar.image("./photo/logo.png", width=200)
# 侧边栏
import base64


def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


your_base64_logo = image_to_base64("./photo/顶部栏.png")
st.sidebar.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{your_base64_logo}" width="30"/>
        <span style="font-size: 27px; font-weight: bold; margin-left: 10px;">Hydrogen Cell LifeX</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# 初始化 session_state
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"  # 默认页面
# 添加自定义 CSS 来放大按钮
st.markdown(
    """
    <style>
        div.stButton > button {
            width: 100%; 
            height: 50px; 
            font-size: 25px;
            background-color: #f0f2f6;  
            font-weight: bold;  /* 加粗 */
            border-radius: 20px;  /* 圆角按钮 */
            border: none;
        }
        /* 悬停效果*/
        div.stButton > button:hover {
            background-color: #e3e5e7;  /* 悬停时变深 */
            opacity: 0.85;  /* 轻微透明度变化 */
            color: initial !important;
        }   
        /* 被点击时的效果 */
        div.stButton > button:focus {
            background-color: #e3e5e7; 
            color: initial !important;
        }

    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
        .appview-container {
            background-color: #f9f9f9;  
        }
        header {
            background-color: #f9f9f9!  important; 
        }
    </style>
    """,
    unsafe_allow_html=True
)
# 侧边栏按钮
if st.sidebar.button("🏠 Home"):
    st.session_state.page = "🏠 Home"

if st.sidebar.button("⚙️ 功能"):
    st.session_state.page = "⚙️ 功能"

# 首页内容
if st.session_state.page == "🏠 Home":

    # 自定义 CSS 样式，设置图片和标题居中
    css = """
    <style>
        /* 使标题居中 */
        h1 {
            text-align: center;
            font-size: 30px;
        }
    </style>
    """
    # 应用自定义 CSS 样式
    st.markdown(css, unsafe_allow_html=True)
    st.image("./photo/图片1.png", width=200)  
    st.title("Hydrogen Cell LifeX")

    st.write("""
        **欢迎使用Hydrogen Cell LifeX！**  

        本软件是一种基于降维分类的氢燃料电池寿命预测软件，旨在解决目前氢燃料电池寿命预测领域单一模型对复杂多变工况数据预测能力不足的问题。模型思路：首先通过PCA降维处理不同的数据集，形成一个综合的相似性图谱。通过这个图谱，我们可以清晰地识别出不同工况下的氢燃料电池的特征表现，并进一步比较它们之间的相似性。在此基础上，当引入新的数据时，系统可以通过与已有图谱进行比对，找到与新数据最为相似的工况，并使用对应的数据集模型进行预测。

        目前，我们已经收集了三组数据集进行进行降维，形成一个相似性图谱。经过初步测试，模型在新数据上的预测结果表现良好，验证了我们的思路和方法的有效性。
    """)

    st.write("""
        **Hydrogen Cell Life功能介绍！**  
        - **功能区**： 功能区提供了一文件上传的按钮
        - **箱线图分析**：用于可视化上传的新数据数据分布情况  
        - **PCA降维相似性分析**：对新数据进行降维，再通过与已有图谱进行比对，找到与新数据最为相似的工况
        - **模型预测结果分析**：使用对应模型进行预测

        请在左侧选择“功能”以进入分析界面。
    """)

    st.write("""
        **Hydrogen Cell Life的示例文件！**  
        https://github.com/withand123/Hydrogen-Cell-LifeX/tree/main/sample_data
    """)
    # 创建按钮

    # if st.button("点击打开链接"):
    #     url = "https://numpy.org/"
    #     # 尝试打开链接
    #     try:
    #         webbrowser.open_new_tab(url)
    #     except Exception as e:
    #         st.error(f"打开链接时出现错误: {e}")

# 功能界面
elif st.session_state.page == "⚙️ 功能":
    if st.sidebar.button("📊 箱线图分析"):
        st.session_state.selected_function = "boxplot"

    if st.sidebar.button("📉 PCA降维相似性分析"):
        st.session_state.selected_function = "pca"

    if st.sidebar.button("🔍 模型预测结果分析"):
        st.session_state.selected_function = "model"
    # 初始化 session_state
    if "selected_function" not in st.session_state:
        st.session_state.selected_function = None
    if "df" not in st.session_state:
        st.session_state.df = None
    # **初始化 session_state**
    if "fig1" not in st.session_state:
        st.session_state.fig1 = None  # 存储当前图像
    if "closest_dataset" not in st.session_state:
        st.session_state.closest_dataset = None

    # 显示 Logo
    st.image("./photo/图片1.png", width=200)
    # 页面标题
    st.title("Hydrogen Cell LifeX")
    # 自定义页面样式
    st.markdown("""
        <style>
        .reportview-container {
            background-color: #f5f5f5;  /* 设置页面背景色 */
        }
        .sidebar .sidebar-content {
            background-color: #2f2f2f;  /* 设置侧边栏背景色 */
            color: white;               /* 设置侧边栏文本颜色 */
        }
        .css-1n6gftb {  /* 针对按钮的样式 */
            background-color: #ff5733 !important;  /* 设置按钮背景色 */
            color: white !important;               /* 设置按钮文字颜色 */
            font-size: 16px !important;            /* 设置按钮字体大小 */
            border-radius: 10px !important;        /* 设置按钮圆角 */
        }
        </style>
        """, unsafe_allow_html=True)
    # 文件上传
    uploaded_file = st.file_uploader("选择文件", type=["csv", "xlsx"])

    if uploaded_file is not None:
        file_name = uploaded_file.name
        if file_name.endswith(".csv"):
            st.session_state.df = pd.read_csv(uploaded_file, index_col=0)
        elif file_name.endswith(".xlsx"):
            st.session_state.df = pd.read_excel(uploaded_file, index_col=0)
        else:
            st.error("不支持的文件格式")

        if st.session_state.df is not None:
            st.write("数据预览：")
            st.dataframe(st.session_state.df)

    # 主界面显示区
    st.subheader("数据可视化区域")

    # 处理不同的功能
    if st.session_state.selected_function == "boxplot":
        st.write("### 箱线图分析")

        if st.session_state.df is not None:
            def xiangxian(data):
                # 确保数据中包含 'Time (h)' 这一列
                from matplotlib import pyplot as plt
                from pylab import mpl
                data.index = data.pop('Time (h)')
                # 设置绘图风格
                sns.set(style="whitegrid", font_scale=1.2)  # 适当增大字体比例
                mpl.font_manager.fontManager.addfont('./font/SimHei.ttf') 
                mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
                mpl.rcParams['axes.unicode_minus'] = False  # 负号- 显示方块
                # 定义颜色调色板
                palette = sns.color_palette("Set2")
                # 创建一个画布，绘制大图箱线图
                fig, ax = plt.subplots(figsize=(12, 6))  # 调整大图尺寸
                ax = plt.gca()
                sns.boxplot(data=data, ax=ax, palette=palette)
                # ax_big.set_title('变量箱线图')
                ax.set_xlabel('变量', fontsize = 16)
                ax.set_ylabel('范围', fontsize = 16)
                ax.tick_params(axis='both', labelsize=16)  # 调整刻度标签字体大小
                # 调整布局并显示图形
                plt.tight_layout()

                st.session_state.fig1 = fig
                st.session_state.file_name = "boxplot.png"  # 设置文件名
                st.pyplot(st.session_state.fig1)

                # 异常值处理函数
                def count_outliers(dff):
                    outliers_count = {}
                    for column in dff.select_dtypes(include=['float64', 'int64']).columns:
                        Q1 = dff[column].quantile(0.25)
                        Q3 = dff[column].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound_15 = Q1 - 1.5 * IQR
                        upper_bound_15 = Q3 + 1.5 * IQR
                        lower_bound_3 = Q1 - 3 * IQR
                        upper_bound_3 = Q3 + 3 * IQR

                        # 温和异常值：落在[lower_bound_3, lower_bound_15) 或 (upper_bound_15, upper_bound_3]
                        mild_outliers = dff[(dff[column] < lower_bound_15) & (dff[column] >= lower_bound_3) |
                                            (dff[column] > upper_bound_15) & (dff[column] <= upper_bound_3)]
                        # 极端异常值：小于 lower_bound_3 或 大于 upper_bound_3
                        extreme_outliers = dff[(dff[column] < lower_bound_3) | (dff[column] > upper_bound_3)]

                        outliers_count[column] = {
                            'mild': len(mild_outliers),
                            'extreme': len(extreme_outliers)
                        }
                    return outliers_count

                def process_outliers(dff):
                    outliers = count_outliers(dff)
                    processed_df = dff.copy()
                    for column in outliers.keys():
                        Q1 = dff[column].quantile(0.25)
                        Q3 = dff[column].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound_3 = Q1 - 3 * IQR
                        upper_bound_3 = Q3 + 3 * IQR
                        # 删除极端异常值
                        processed_df = processed_df[
                            (processed_df[column] >= lower_bound_3) & (processed_df[column] <= upper_bound_3)]
                    return processed_df, outliers

                # 处理异常值
                cleaned_data, outlier_stats = process_outliers(data)
                # 显示异常值统计信息
                st.write("### 异常值统计")
                st.write(pd.DataFrame(outlier_stats).T)

                # 恢复时间列
                processed_data = cleaned_data.reset_index().rename(columns={'index': 'Time (h)'})
                # 将处理后的数据传回 `st.session_state.df`
                st.session_state.df = processed_data

                # 确认数据已更新
                st.success("异常值已处理，数据已更新！")


            xiangxian(st.session_state.df)
        else:
            st.warning("请先上传数据！")

    elif st.session_state.selected_function == "pca":
        st.write("### PCA 降维相似性分析")
        if st.session_state.df is not None:
            # PCA效果展现    加入一个判断靠经那个model
            def pca_effect(data4):
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                from matplotlib import pyplot as plt
                from pylab import mpl
                import numpy as np
                mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
                mpl.rcParams['axes.unicode_minus'] = False  # 负号- 显示方块
                import joblib
                # 加载降维后的数据
                data_pca = np.load('./DefaultData/data_pca.npy')

                # 加载标准化器
                scaler = joblib.load('./DefaultData/pca_scaler.joblib')

                # 加载PCA模型
                pca = joblib.load('./DefaultData/pca_model.joblib')
                data_scaled4 = scaler.transform(data4)
                # 使用相同的PCA模型对新数据进行变换
                new_pca_result4 = pca.transform(data_scaled4)
                labels = ['FC数据集'] * 102974 + ['国家数据中心数据集'] * 795044 + ['同济大学数据集'] * 1222746
                # 定义颜色映射
                colors = {'FC数据集': 'red', '国家数据中心数据集': 'blue', '同济大学数据集': 'green'}
                # 获取解释方差比
                explained_variance_ratio = pca.explained_variance_ratio_
                import numpy as np
                # 可视化
                fig, ax = plt.subplots(figsize=(10, 8))
                for label in set(labels):
                    mask = np.array(labels) == label
                    plt.scatter(data_pca[mask, 0], data_pca[mask, 1], color=colors[label], label=label)
                labels1 = ['新数据集1'] * len(data4)
                mask1 = np.array(labels1) == '新数据集1'
                plt.scatter(new_pca_result4[mask1, 0], new_pca_result4[mask1, 1], label='新数据集1')
                plt.xlabel(f"PC1 ({explained_variance_ratio[0] * 100:.2f}%)", fontsize=16)
                plt.ylabel(f"PC2 ({explained_variance_ratio[1] * 100:.2f}%)", fontsize=16)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.legend(loc='upper left', prop={'size': 16})

                st.session_state.fig1 = fig
                st.session_state.file_name = "pca_plot.png"  # 设置文件名
                st.pyplot(st.session_state.fig1)

                # 计算data4与各个数据集的平均距离
                # 将数据集分组
                fc_mask = np.array(labels) == 'FC数据集'
                guojia_mask = np.array(labels) == '国家数据中心数据集'
                tongji_mask = np.array(labels) == '同济大学数据集'

                # 计算每个数据集在PCA空间中的质心
                fc_centroid = np.mean(data_pca[fc_mask], axis=0)
                guojia_centroid = np.mean(data_pca[guojia_mask], axis=0)
                tongji_centroid = np.mean(data_pca[tongji_mask], axis=0)

                # 计算data4在PCA空间中的质心
                data4_centroid = np.mean(new_pca_result4, axis=0)

                # 计算与各个数据集质心的欧氏距离
                dist_to_fc = np.linalg.norm(data4_centroid - fc_centroid)
                dist_to_guojia = np.linalg.norm(data4_centroid - guojia_centroid)
                dist_to_tongji = np.linalg.norm(data4_centroid - tongji_centroid)

                # 找出最近的数据集
                distances = {
                    'FC数据集': dist_to_fc,
                    '国家数据中心数据集': dist_to_guojia,
                    '同济大学数据集': dist_to_tongji
                }

                st.session_state.closest_dataset = min(distances, key=distances.get)
                # 确认数据已更新
                st.success(f"该数据集靠近{st.session_state.closest_dataset}！")


            pca_effect(st.session_state.df)
        else:
            st.warning("请先上传数据！")
    elif st.session_state.selected_function == "model":
        st.write("### 模型预测结果分析")
        if st.session_state.closest_dataset is not None:
            # 模型预测效果
            def model_effect(closest_dataset, data):
                import pandas as pd
                import numpy as np
                import os
                os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
                import tensorflow as tf
                from tensorflow.keras.models import load_model
                # 根据closest_dataset判断使用哪个模型
                if closest_dataset == 'FC数据集':
                    def kan_transform(x):
                        return tf.square(x) * 0.45

                    model = load_model('./model/FC_kan_lstm_model.keras',
                                       custom_objects={'kan_transform': kan_transform})
                elif closest_dataset == '国家数据中心数据集':
                    def kan_transform(x):
                        return tf.square(x) * 0.395

                    model = load_model('./model/guojia_kan_lstm_model.keras',
                                       custom_objects={'kan_transform': kan_transform})
                elif closest_dataset == '同济大学数据集':
                    def kan_transform(x):
                        return tf.square(x) * 0.45

                    # 加载模型时指定 custom_objects
                    model = load_model('./model/tongji_kan_lstm_model.keras',
                                       custom_objects={'kan_transform': kan_transform})
                from sklearn import preprocessing
                min_max_scalar = preprocessing.MinMaxScaler()
                step = 1
                ahead = 1

                def createXY(dataset, n_pase, step_ahead):
                    dataX = []
                    dataY = []
                    for i in range(n_pase, len(dataset) - step_ahead):
                        dataX.append(dataset[i - n_pase:i, 0:dataset.shape[1]])
                        dataY.append(dataset[i + step_ahead, 0])
                    return np.array(dataX), np.array(dataY)

                # 读入新数据
                # sampled_data1 = pd.read_csv('generated_df7.csv', index_col=0)
                # # 生成从 1 到 1000H 的时间标签
                # time_labels = [f'{i}' for i in range(1, len(sampled_data1) + 1)]
                # # 将时间列替换为新的时间标签
                # sampled_data1['Time (h)'] = time_labels
                # import statsmodels.api as sm
                # data_1 = pd.read_csv('new_data7.csv', index_col=0)
                # 获取时间列的最大值 c
                max_time = data['Time (h)'].max()
                # 定义时间区间（从0到最大时间，每小时一个区间）
                time_intervals = [(i, i + 1) for i in range(int(max_time) + 1)]
                # 存储每个时间区间中随机选择的一行数据
                sampled_data = pd.DataFrame()
                for start, end in time_intervals:
                    # 从时间区间[start, end)中筛选数据
                    interval_data = data[(data['Time (h)'] >= start) & (data['Time (h)'] < end)]

                    if not interval_data.empty:
                        # 随机选择一行数据
                        sampled_row = interval_data.sample(n=1)
                        sampled_data = pd.concat([sampled_data, sampled_row], ignore_index=True)
                # 设置 LOESS 平滑的参数
                window_width = 20
                window_width1 = 25
                import matplotlib.pyplot as plt
                from pylab import mpl
                import seaborn as sns
                mpl.rcParams['font.sans-serif'] = ['Times New Roman']  # 指定默认字体
                mpl.rcParams['axes.unicode_minus'] = False  # 负号- 显示方块
                from matplotlib.font_manager import FontProperties
                simsun = FontProperties(family='SimHei', size=16)
                # 使用 LOESS 平滑数据
                lowess = sm.nonparametric.lowess(sampled_data['Utot (V)'], sampled_data['Time (h)'],
                                                 frac=window_width / len(sampled_data))
                lowess1 = sm.nonparametric.lowess(sampled_data['I (A)'], sampled_data['Time (h)'],
                                                  frac=window_width1 / len(sampled_data))
                lowess2 = sm.nonparametric.lowess(sampled_data['PoutH2 (Kpa)'], sampled_data['Time (h)'],
                                                  frac=window_width1 / len(sampled_data))
                sampled_data['I (A)'] = lowess1[:, 1]
                sampled_data['Utot (V)'] = lowess[:, 1]
                sampled_data['PoutH2 (Kpa)'] = lowess2[:, 1]
                sampled_data.index = sampled_data.pop('Time (h)')
                data_for_testing_scaled1 = min_max_scalar.fit_transform(sampled_data)
                tryX, tryY = createXY(data_for_testing_scaled1, step, ahead)
                prediction_try = model.predict(tryX)
                prediction_copied_array_try = np.repeat(prediction_try, 8, axis=-1)
                pred_try = min_max_scalar.inverse_transform(
                    np.reshape(prediction_copied_array_try, (len(prediction_try), 8)))[:, 0]
                original_copies_array_try = np.repeat(tryY, 8, axis=-1)
                original_try = min_max_scalar.inverse_transform(np.reshape(original_copies_array_try, (len(tryY), 8)))[
                               :, 0]
                # data_time = train.index[-len(original):]
                data_time_try = len(original_try)
                from matplotlib import pyplot as plt
                from pylab import mpl
                mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
                mpl.rcParams['axes.unicode_minus'] = False  # 负号- 显示方块
                from matplotlib import pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 8))  # 调整大图尺寸
                plt.plot(range(data_time_try), original_try, color='c', label='实际值')
                plt.plot(range(data_time_try), pred_try, color='red', label='FC数据集模型')
                plt.xlabel('时间(h)')
                plt.ylabel('电堆电压(V)')

                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.legend(loc='upper left', prop={'size': 16})
                st.session_state.fig1 = fig
                st.session_state.file_name = "model_prediction.png"  # 设置文件名
                st.pyplot(fig)
                from sklearn import metrics
                import numpy as np
                # 评价
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                import math
                mse = mean_squared_error(np.array(pred_try), np.array(original_try))
                # calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
                rmse = math.sqrt(mean_squared_error(pred_try, original_try))
                # calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
                mae = mean_absolute_error(pred_try, original_try)

                def mape(actual_arr, pred_arr):
                    mask = np.abs(actual_arr) > 0.001
                    actual_arr = actual_arr[mask]
                    pred_arr = pred_arr[mask]
                    mape = np.mean(abs(((actual_arr - pred_arr) / actual_arr)))
                    return mape

                mape = mape(original_try, pred_try)
                st.write('决定系数R2: %.6f' % metrics.r2_score(original_try, pred_try))
                st.write('均方误差MSE: %.6f' % mse)
                st.write('均方根误差RMSE: %.6f' % rmse)
                st.write('平均绝对误差MAE: %.6f' % mae)
                st.write('平均绝对百分比误差MAPE: %.6f' % mape)


            model_effect(st.session_state.closest_dataset, st.session_state.df)
        else:
            st.warning("请先进行PCA降维相似性分析！")

    # **通用导出按钮**
    if st.session_state.fig1 is not None:
        buffer = io.BytesIO()
        st.session_state.fig1.savefig(buffer, format="png")
        buffer.seek(0)

        st.download_button(
            label="📥 导出图片",
            data=buffer,
            file_name=st.session_state.file_name,
            mime="image/png"
        )
