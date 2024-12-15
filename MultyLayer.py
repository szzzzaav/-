import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import re
from tqdm import tqdm
import matplotlib.pyplot as plt


class AnswerScorer:
    def __init__(self, epochs=100):
        print("\n初始化模型...")
        self.epochs = epochs
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            analyzer='word',
            max_features=5000,  # 增加特征数量到5000
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.9
        )
        self.scaler = StandardScaler()
        self.classifier = MLPClassifier(
            # 增加网络层数和每层神经元数量
            hidden_layer_sizes=(512, 256, 128, 64),
            max_iter=1,
            warm_start=True,
            # 使用学习率调度策略
            learning_rate='adaptive',
            learning_rate_init=0.001,
            # 增加batch size
            batch_size=128,
            # 添加dropout
            early_stopping=True,
            validation_fraction=0.1,
            # L2正则化参数
            alpha=0.0005,
            # 使用ELU激活函数
            activation='relu',
            solver='adam',
            # 添加动量
            momentum=0.9,
            random_state=42
        )
        print("模型初始化完成！")

    def preprocess_text(self, text):
        """预处理文本"""
        if not isinstance(text, str):
            text = str(text)
        # 移除特殊字符，保留数字和字母
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.lower().strip()

    def extract_features(self, student_answer, reference_answer):
        """提取特征"""
        # 预处理文本
        student_answer = self.preprocess_text(student_answer)
        reference_answer = self.preprocess_text(reference_answer)

        # 计算TF-IDF向量
        tfidf_matrix = self.vectorizer.fit_transform([student_answer, reference_answer])

        # 计算余弦相似度
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # 计算长度比率
        len_ratio = len(student_answer) / (len(reference_answer) + 1)  # 加1避免除零

        # 计算数值匹配度（如果答案包含数字）
        student_numbers = set(re.findall(r'\d+(?:\.\d+)?', student_answer))
        ref_numbers = set(re.findall(r'\d+(?:\.\d+)?', reference_answer))
        number_match = len(student_numbers.intersection(ref_numbers)) / (len(ref_numbers) + 1)

        # 计算关键词匹配度
        student_words = set(student_answer.split())
        ref_words = set(reference_answer.split())
        word_match = len(student_words.intersection(ref_words)) / (len(ref_words) + 1)

        # 计算词序相似度
        student_words_list = student_answer.split()
        ref_words_list = reference_answer.split()
        sequence_match = 0
        if len(ref_words_list) > 0:
            from difflib import SequenceMatcher
            sequence_match = SequenceMatcher(None, student_words_list, ref_words_list).ratio()

        return np.array([cosine_sim, len_ratio, number_match, word_match, sequence_match])

    def preprocess_data(self, data):
        """处理数据集"""
        print("\n开始特征提取...")
        features = []
        for item in tqdm(data, desc="处理数据"):
            feat = self.extract_features(item['answer'], item['reference_answer'])
            features.append(feat)
        print("特征提取完成！")
        return np.array(features)

    def train(self, train_data):
        """训练模型并评估"""
        print("\n开始数据预处理...")
        # 划分训练集和验证集
        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
        print(f"训练集大小: {len(train_data)}")
        print(f"验证集大小: {len(val_data)}")

        # 准备训练数据
        print("\n处理训练集...")
        X_train = self.preprocess_data(train_data)
        y_train = np.array([1 if 'score' in item and item['score'] == 1 else 0 for item in train_data])

        # 准备验证数据
        print("\n处理验证集...")
        X_val = self.preprocess_data(val_data)
        y_val = np.array([1 if 'score' in item and item['score'] == 1 else 0 for item in val_data])

        # 标准化特征
        print("\n特征标准化...")
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        # 训练模型
        print(f"\n开始训练模型... (最大轮数: {self.epochs})")

        # 记录每轮的评估指标
        history = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

        # 初始化分类器
        self.classifier = MLPClassifier(
            hidden_layer_sizes=(200, 100, 50),
            max_iter=1,  # 每次只训练一轮
            warm_start=True,  # 启用热启动，在前一次训练的基础上继续
            learning_rate_init=0.0001,
            batch_size=64,
            alpha=0.0001,
            activation='relu',
            solver='adam',
            random_state=42
        )

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print("-" * 30)

            # 训练一轮
            self.classifier.fit(X_train, y_train)

            # 在验证集上评估
            val_predictions = self.classifier.predict(X_val)

            # 计算评估指标
            accuracy = accuracy_score(y_val, val_predictions)
            precision = precision_score(y_val, val_predictions, average='weighted')
            recall = recall_score(y_val, val_predictions, average='weighted')
            f1 = f1_score(y_val, val_predictions, average='weighted')

            # 记录评估指标
            history['accuracy'].append(accuracy)
            history['precision'].append(precision)
            history['recall'].append(recall)
            history['f1'].append(f1)

            print(f"验证集评估结果:")
            print(f"准确率 (Accuracy): {accuracy:.4f}")
            print(f"精确率 (Precision): {precision:.4f}")
            print(f"召回率 (Recall): {recall:.4f}")
            print(f"F1分数 (F1 Score): {f1:.4f}")

        print("\n训练完成!")

        # 绘制评估指标图表
        plt.figure(figsize=(12, 8))
        epochs_range = range(1, self.epochs + 1)

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        # 绘制四个指标的曲线
        plt.plot(epochs_range, history['accuracy'], label='准确率', marker='o')
        plt.plot(epochs_range, history['precision'], label='精确率', marker='s')
        plt.plot(epochs_range, history['recall'], label='召回率', marker='^')
        plt.plot(epochs_range, history['f1'], label='F1分数', marker='D')

        plt.title('模型训练过程中的评估指标变化')
        plt.xlabel('训练轮次')
        plt.ylabel('指标值')
        plt.grid(True)
        plt.legend()

        # 找出最佳轮次
        best_epoch = np.argmax(history['f1']) + 1
        best_metrics = {
            '准确率': history['accuracy'][best_epoch - 1],
            '精确率': history['precision'][best_epoch - 1],
            '召回率': history['recall'][best_epoch - 1],
            'F1分数': history['f1'][best_epoch - 1]
        }

        print("\n最佳型指标 (第{}轮):".format(best_epoch))
        print("=" * 50)
        for metric, value in best_metrics.items():
            print(f"{metric}: {value:.4f}")
        print("=" * 50)

        # 保存图表
        plt.savefig('training_metrics.png')
        print("\n评估指标图表已保存为 'training_metrics.png'")
        plt.close()

    def predict(self, test_data):
        """预测"""
        print("\n开始生成预测...")
        X = self.preprocess_data(test_data)
        X = self.scaler.transform(X)
        predictions = self.classifier.predict(X)
        print("预测完成！")
        return predictions


def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse line: {line}")
                        continue
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    return data


def main():

    # 取训练数据
    print("\n读取训练数据...")
    train_data = load_jsonl('train.jsonl')
    if train_data is None:
        print("错误：无法加载训练数据")
        return
    print(f"成功加载训练数据：{len(train_data)}条记录")

    # 读取测试数据
    print("\n读取测试数据...")
    test_data = load_jsonl('test.jsonl')
    if test_data is None:
        print("错误：无法加载测试数据")
        print("Failed to load test data")
        return

    # 初始化评分器（设置epochs）
    scorer = AnswerScorer(epochs=100)

    # 训练模型并评估
    scorer.train(train_data)

    # 预测测试集
    predictions = scorer.predict(test_data)

    # 生成提交文件
    results = pd.DataFrame({
        'question_id': [item['question_id'] for item in test_data],
        'student_id': [item['student_id'] for item in test_data],
        'score': predictions
    })

    # 保存结果
    try:
        results.to_csv('group1.csv', index=False, encoding='utf-8')
        print("\nPredictions saved successfully")
    except Exception as e:
        print(f"\nError saving predictions: {e}")


if __name__ == '__main__':
    main()
