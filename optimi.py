import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

EPOCHS = 32


class AnswerScorer:
    def __init__(self, epochs=100):
        print("\n初始化模型...")
        self.epochs = epochs
        # 优化TF-IDF向量化器参数
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            analyzer='word',
            max_features=5000,
            ngram_range=(1, 3),  # 使用1-3gram特征
            min_df=2,  # 最小文档频率
            max_df=0.95  # 最大文档频率
        )
        self.scaler = StandardScaler()
        # 优化MLPClassifier参数
        self.classifier = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),  # 优化网络结构
            max_iter=1,
            warm_start=True,
            learning_rate='adaptive',
            learning_rate_init=0.0005,  # 降低学习率
            batch_size=32,  # 减小batch size
            validation_fraction=0.2,
            alpha=0.001,  # 增加正则化强度
            activation='relu',
            solver='adam',
            momentum=0.9,
            random_state=42
        )
        print("模型初始化完成！")

    def preprocess_text(self, text):
        """预处理文本"""
        if not isinstance(text, str):
            text = str(text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.lower().strip()

    def preprocess_data(self, data):
        """处理数据集"""
        print("\n开始特征提取...")
        features = []
        for item in tqdm(data, desc="处理数据"):
            # 提取特征
            feat = self.extract_features(item['answer'], item['reference_answer'])
            features.append(feat)
        print("特征提取完成！")
        return np.array(features)

    def extract_features(self, student_answer, reference_answer):
        """增强的特征提取"""
        # 预处理文本
        student_answer = self.preprocess_text(student_answer)
        reference_answer = self.preprocess_text(reference_answer)

        # 修改vectorizer的参数
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            analyzer='word',
            max_features=5000,
            ngram_range=(1, 3),
            min_df=1,  # 改为1，因为我们只比较两个文档
            max_df=1.0  # 改为1.0，允许所有词项
        )

        # 计算TF-IDF向量
        tfidf_matrix = self.vectorizer.fit_transform([student_answer, reference_answer])

        # 计算余弦相似度
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # 计算长度比率
        len_ratio = len(student_answer) / (len(reference_answer) + 1)

        # 计算数值匹配度
        student_numbers = set(re.findall(r'\d+(?:\.\d+)?', student_answer))
        ref_numbers = set(re.findall(r'\d+(?:\.\d+)?', reference_answer))
        number_match = len(student_numbers.intersection(ref_numbers)) / (len(ref_numbers) + 1)

        # 计算关键词匹配度
        student_words = set(student_answer.split())
        ref_words = set(reference_answer.split())
        word_match = len(student_words.intersection(ref_words)) / (len(ref_words) + 1)

        # 计算词序相似度
        sequence_match = 0
        if len(ref_words) > 0:
            from difflib import SequenceMatcher
            sequence_match = SequenceMatcher(None, student_answer.split(), reference_answer.split()).ratio()

        # 添加新特征：词长度差异
        avg_word_len_diff = abs(
            np.mean([len(w) for w in student_words]) -
            np.mean([len(w) for w in ref_words])
        ) if student_words and ref_words else 0

        return np.array([
            cosine_sim,
            len_ratio,
            number_match,
            word_match,
            sequence_match,
            avg_word_len_diff
        ])

    def train(self, train_data):
        """增强的训练过程"""
        print("\n开始数据预处理...")
        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
        print(f"训练集大小: {len(train_data)}")
        print(f"验证集大小: {len(val_data)}")

        # 准备数据
        X_train = self.preprocess_data(train_data)
        y_train = np.array([1 if 'score' in item and item['score'] == 1 else 0 for item in train_data])
        X_val = self.preprocess_data(val_data)
        y_val = np.array([1 if 'score' in item and item['score'] == 1 else 0 for item in val_data])

        # 标准化特征
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        # 记录训练历史
        history = {
            'train_accuracy': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }

        print(f"\n开始训练模型... (最大轮数: {self.epochs})")
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print("-" * 30)

            # 训练
            self.classifier.fit(X_train, y_train)

            # 评估训练集
            train_pred = self.classifier.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_pred)

            # 评估验证集
            val_pred = self.classifier.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            val_precision = precision_score(y_val, val_pred, average='weighted')
            val_recall = recall_score(y_val, val_pred, average='weighted')
            val_f1 = f1_score(y_val, val_pred, average='weighted')

            # 记录指标
            history['train_accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)
            history['val_precision'].append(val_precision)
            history['val_recall'].append(val_recall)
            history['val_f1'].append(val_f1)

            print(f"训练集准确率: {train_accuracy:.4f}")
            print(f"验证集评估结果:")
            print(f"准确率: {val_accuracy:.4f}")
            print(f"精确率: {val_precision:.4f}")
            print(f"召回率: {val_recall:.4f}")
            print(f"F1分数: {val_f1:.4f}")

        print("\n训练完成!")

        # 绘制训练过程指标
        self.plot_training_metrics(history)

    def plot_training_metrics(self, history):
        """绘制完整的训练指标"""
        plt.figure(figsize=(15, 10))

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 绘制准确率对比
        plt.subplot(2, 1, 1)
        plt.plot(history['train_accuracy'], label='训练准确率', marker='o')
        plt.plot(history['val_accuracy'], label='验证准确率', marker='s')
        plt.title('模型准确率随训练轮次的变化')
        plt.xlabel('训练轮次')
        plt.ylabel('准确率')
        plt.grid(True)
        plt.legend()

        # 绘制验证集的所有指标
        plt.subplot(2, 1, 2)
        plt.plot(history['val_accuracy'], label='准确率', marker='o')
        plt.plot(history['val_precision'], label='精确率', marker='s')
        plt.plot(history['val_recall'], label='召回率', marker='^')
        plt.plot(history['val_f1'], label='F1分数', marker='D')
        plt.title('验证集评估指标随训练轮次的变化')
        plt.xlabel('训练轮次')
        plt.ylabel('指标值')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_metrics.png')
        print("\n评估指标图表已保存为 'training_metrics.png'")
        plt.close()

    def predict(self, test_data):
        """预测"""
        print("\n开始生成预测...")
        X = self.preprocess_data(test_data)
        predictions = (self.model.predict(X) > 0.5).astype("int32")
        print("预测完成！")
        return predictions.flatten()


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
        return

    # 初始化评分器（设置epochs）
    scorer = AnswerScorer(epochs=EPOCHS)

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
