
i# review_analyzer_core.py (版本 2.0 - 适配“基础+覆写”逻辑)

import nltk
import pandas as pd
import numpy as np
from textblob import TextBlob
from typing import List, Dict, Any
import json
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from collections.abc import Mapping
import copy

class ReviewAnalyzer:
    """
    一个用于处理和分析产品评论数据的可复用工具。
    它封装了完整的分析流程，通过配置驱动，易于扩展和复用。 (V6.1 引擎)
    """

    def __init__(self, config: Dict, product_type: str = "standard"):
        """
        【V9.1 基础+覆写版】
        初始化分析器。此版本专门设计用于处理“基础”关键词和特定产品画像的“覆写”规则。
        """
        self.config = config
        self.df = None
        self.product_type = product_type

        print(f"正在为【{self.product_type}】产品创建一个专属分析器...")

        # 1. 加载所有基础和画像规则
        self._load_all_keywords()

        # 2. 根据 product_type，动态生成最终的关键词词库
        final_keywords = self._get_keywords_for_product(self.product_type)

        # 3. 【核心操作】用最终生成的词库，覆盖掉原始配置中的词库
        self.config['feature_keywords'] = final_keywords
        print("✅ 专属关键词词库已生成并注入配置！后续所有分析将使用此定制规则。")

        # 4. 执行NLTK资源初始化
        self._initialize_nltk_resources()

    def _load_all_keywords(self):
        """
        在初始化时，从配置中加载“基础”关键词和所有产品“画像”。
        """
        # 从主配置中获取基础关键词
        self.config['base_keywords'] = self.config.get('base_keywords', {})
        # 从主配置中获取所有画像（覆写规则）
        self.config['profiles'] = self.config.get('profiles', {})

        loaded_profiles = list(self.config['profiles'].keys())
        if loaded_profiles:
            print(f"已加载 {len(loaded_profiles)} 个产品画像: {loaded_profiles}")
        else:
            print("警告: 未在配置中找到任何产品画像。")

    def _deep_merge_dicts(self, d1, d2):
        """
        递归地合并两个字典。d2中的值会覆写d1中的值。
        如果d2中的列表为空，它会清空d1中对应的列表。
        """
        for k, v in d2.items():
            if k in d1 and isinstance(d1[k], dict) and isinstance(v, Mapping):
                self._deep_merge_dicts(d1[k], v)
            else:
                d1[k] = v
        return d1

    def _get_keywords_for_product(self, product_type: str) -> dict:
        """
        根据产品类型，通过“基础+覆写”的逻辑，动态生成最终的关键词词库。
        """
        # 1. 深度复制一份基础规则，作为起点
        final_keywords = copy.deepcopy(self.config['base_keywords'])

        # 2. 检查是否存在该产品类型的专属画像（覆写规则）
        if product_type in self.config.get('profiles', {}):
            print(f"检测到产品类型 '{product_type}', 正在应用专属画像...")
            profile_overrides = self.config['profiles'][product_type]
            
            # 3. 将覆写规则深度合并到基础规则上
            final_keywords = self._deep_merge_dicts(final_keywords, profile_overrides)
        else:
            print(f"未找到产品 '{product_type}' 的专属画像，将仅使用基础规则。")

        return final_keywords

    def _initialize_nltk_resources(self):
        """一次性检查并下载所有需要的NLTK数据包。"""
        print("正在检查NLTK资源...")
        required_packages = {'tokenizers/punkt': 'punkt', 'corpora/stopwords': 'stopwords', 'corpora/wordnet': 'wordnet'}
        for path, package_id in required_packages.items():
            try:
                nltk.data.find(path)
            except LookupError:
                print(f"正在下载NLTK包: {package_id}...")
                nltk.download(package_id, quiet=True)
        print("NLTK资源准备就绪。")

    def _preprocess_text(self, text: str) -> str:
        """执行完整的文本预处理：小写、去特殊字符、分词、去停用词、词形还原。"""
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [
            lemmatizer.lemmatize(word) for word in tokens
            if word not in stop_words and len(word) > 2
        ]
        return ' '.join(lemmatized_tokens)


    def _precompute_feature_sentiments(self):
        """
        【V8.2 黄金最终版：“解耦”引擎】
        - 步骤一：使用高效的矢量化操作，并结合“全词匹配”，精准判断所有特征的“提及”。
        - 步骤二：仅针对被提及的评论，启动“句子级情感分析”，精准判断情感。
        - 彻底解决了新旧引擎逻辑冲突导致结果无变化的问题。
        """
        print("\n正在启动【V8.2 黄金最终版引擎】进行预计算...")
        feature_keywords_config = self.config.get('feature_keywords', {})
        if not feature_keywords_config:
            return

        content_col = self.config['content_column']
        rating_col = self.config['rating_column']

        if 'Processed_Text' not in self.df.columns:
            self.df['Processed_Text'] = self.df[content_col].apply(self._preprocess_text)

        # --- 步骤一：矢量化、全词匹配的“特征提及”判断 ---
        print(" - 步骤 1/2: 正在高效、精准地判断所有特征提及...")
        for feature, sub_topics in feature_keywords_config.items():
            all_keywords = [kw for kws in sub_topics.values() for kw in kws]
            # 使用全词匹配 r'\b(word1|word2)\b'
            keyword_pattern = r'\b(' + '|'.join(list(set([re.escape(kw) for kw in all_keywords]))) + r')\b'
            self.df[f'feature_{feature}'] = self.df['Processed_Text'].str.contains(keyword_pattern, regex=True, na=False).astype(int)

        # --- 步骤二：逐句分析、精准“情感归因” ---
        print(" - 步骤 2/2: 正在对已提及的特征进行句子级情感归因...")

        # 先初始化所有情感得分列
        for feature in feature_keywords_config.keys():
            self.df[f'sentiment_score_{feature}'] = 0.0

        # 筛选出至少提及一个特征的评论，只对这些评论进行慢速的逐句分析
        mention_cols = [f'feature_{f}' for f in feature_keywords_config.keys()]
        reviews_with_mentions = self.df[self.df[mention_cols].sum(axis=1) > 0]

        for index, row in reviews_with_mentions.iterrows():
            review_text = row[content_col]
            if not isinstance(review_text, str) or pd.isna(review_text):
                continue

            sentences = sent_tokenize(review_text)

            for feature, sub_topics in feature_keywords_config.items():
                # 只分析该条评论确实提及了的特征
                if row[f'feature_{feature}'] == 1:
                    feature_sentiments = []
                    for sentence in sentences:
                        processed_sentence = self._preprocess_text(sentence)
                        sentence_polarity = 0
                        strong_sentiment_found = False

                        # 优先匹配情感化子主题
                        for sub_topic, keywords in sub_topics.items():
                            if not keywords: continue
                            pattern = r'\b(' + '|'.join([re.escape(kw) for kw in keywords]) + r')\b'
                            if re.search(pattern, processed_sentence, re.IGNORECASE):
                                if sub_topic.startswith('正面'):
                                    sentence_polarity = 1.0
                                    strong_sentiment_found = True
                                    break
                                elif sub_topic.startswith('负面'):
                                    sentence_polarity = -1.0
                                    strong_sentiment_found = True
                                    break

                        # 如果没有强情感词，再对中性词句子进行情感分析
                        if not strong_sentiment_found:
                            for sub_topic, keywords in sub_topics.items():
                                if sub_topic.startswith(('正面', '负面')): continue
                                if not keywords: continue
                                pattern = r'\b(' + '|'.join([re.escape(kw) for kw in keywords]) + r')\b'
                                if re.search(pattern, processed_sentence, re.IGNORECASE):
                                    sentence_polarity = TextBlob(sentence).sentiment.polarity
                                    break

                        if sentence_polarity != 0:
                            feature_sentiments.append(sentence_polarity)

                    if feature_sentiments:
                        avg_sentiment = sum(feature_sentiments) / len(feature_sentiments)
                        self.df.loc[index, f'sentiment_score_{feature}'] = avg_sentiment

        # --- 最后一步：根据情感得分，生成最终的情感标签 (1, 0, -1) ---
        for feature in feature_keywords_config.keys():
            score_col = f'sentiment_score_{feature}'
            sentiment_col = f'sentiment_{feature}'
            conditions = [self.df[score_col] > 0.05, self.df[score_col] < -0.05]
            choices = [1, -1]
            self.df[sentiment_col] = np.select(conditions, choices, default=0)

        print("✅ 情感引擎预计算完成！")


    def _load_and_clean_data(self):
        """内部方法：加载并执行基础数据清洗。"""
        try:
            filepath = self.config['input_filepath']
            print(f"正在从 '{filepath}' 加载数据...")
            self.df = pd.read_excel(filepath)
            content_col, rating_col = self.config['content_column'], self.config['rating_column']
            self.df = self.df.dropna(subset=[content_col, rating_col])
            self.df['Content_Clean'] = self.df[content_col].astype(str).str.lower()
            self.df[rating_col] = pd.to_numeric(self.df[rating_col], errors='coerce')
            self.df = self.df.dropna(subset=[rating_col])
            print("数据加载和基础清洗完成。")
            return True
        except FileNotFoundError:
            print(f"错误: 文件 '{self.config['input_filepath']}' 未找到。")
            return False
        except KeyError as e:
            print(f"错误: 配置文件中的列名 {e} 在Excel文件中未找到。")
            return False

    def analyze_sentiment(self):
        """对清洗后的内容进行情感分析。"""
        print("正在进行情感分析...")
        self.df['Sentiment'] = self.df['Content_Clean'].apply(lambda text: TextBlob(text).sentiment.polarity)
        self.df['Sentiment_Category'] = pd.cut(self.df['Sentiment'], bins=self.config['sentiment_bins'], labels=self.config['sentiment_labels'])
        print("情感分析完成。")

    def extract_keywords(self):
        """根据配置的关键词列表提取关键词。"""
        print("正在提取关键词...")
        for keyword in self.config['keywords']:
            self.df[f'has_{keyword}'] = self.df['Content_Clean'].str.contains(keyword, regex=False).astype(int)
        print("关键词提取完成。")

    def categorize_products(self):
        """根据ASIN进行产品分类。"""
        print("正在根据ASIN进行产品分类...")
        asin_col = self.config.get('model_column', 'Asin')
        mapping = {str(k).lower(): v for k, v in self.config['category_mapping'].items()}
        self.df['Product_Category'] = self.df[asin_col].astype(str).str.lower().map(mapping).fillna("Other Series")
        print("产品分类完成。")

    # (请用此版本完整替换旧的 classify_by_rules 函数)
    def classify_by_rules(self, new_column_name: str, classification_key: str, default_value: str = "其他"):
        """
        【V6.5 升级版】一个通用的、由配置驱动的分类方法。
        - 使用正则表达式和单词边界(\b)实现全词匹配，避免子字符串误判。
        """
        print(f"正在使用【全词匹配】模式进行 '{new_column_name}' 分类...")

        rules = self.config.get('classification_rules', {}).get(classification_key, {})
        if not rules:
            print(f"警告: 在配置中未找到 '{classification_key}' 的分类规则。将所有条目设为默认值 '{default_value}'。")

        target_col = self.config['content_column']

        # 【核心修改】为每个分类预编译一个高效的正则表达式
        # 例如，'男性'的模式会变成 r'\b(man|men|boy|...)\b'
        compiled_rules = {
            category: re.compile(r'\b(' + '|'.join(keywords) + r')\b', re.IGNORECASE)
            for category, keywords in rules.items()
        }

        def classifier(text: str) -> str:
            # 这里不再需要 .lower()，因为 re.IGNORECASE 会处理大小写
            text = str(text)
            for category, pattern in compiled_rules.items():
                if pattern.search(text):
                    return category
            return default_value

        self.df[new_column_name] = self.df[target_col].apply(classifier)
        print(f"'{new_column_name}' 分类完成。")

    def generate_feature_analysis_report(self) -> Dict:
        """生成一个关于产品特征的、包含四大部分的完整分析报告。"""
        print("\n正在生成产品优缺点综合分析报告...")
        feature_keywords = self.config.get('feature_keywords', {})
        if not feature_keywords:
            return {}

        feature_sentiment_stats = {}
        for feature in feature_keywords.keys():
            if f'feature_{feature}' not in self.df.columns: continue
            total_mentions = int(self.df[f'feature_{feature}'].sum())
            if total_mentions > 0:
                positive_mentions = int((self.df[f'sentiment_{feature}'] == 1).sum())
                negative_mentions = int((self.df[f'sentiment_{feature}'] == -1).sum())
                feature_sentiment_stats[feature] = {
                    'total_mentions': total_mentions,
                    'positive_ratio': (positive_mentions / total_mentions * 100),
                    'negative_ratio': (negative_mentions / total_mentions * 100)
                }

        high_ratings_df = self.df[self.df['Rating'] >= 4]
        low_ratings_df = self.df[self.df['Rating'] <= 3]
        rating_group_mention_rates = {'high_ratings': {}, 'low_ratings': {}}
        for feature in feature_keywords.keys():
            if f'feature_{feature}' not in self.df.columns: continue
            high_mentions = high_ratings_df[f'feature_{feature}'].sum()
            rating_group_mention_rates['high_ratings'][feature] = (high_mentions / len(high_ratings_df) * 100) if len(high_ratings_df) > 0 else 0
            low_mentions = low_ratings_df[f'feature_{feature}'].sum()
            rating_group_mention_rates['low_ratings'][feature] = (low_mentions / len(low_ratings_df) * 100) if len(low_ratings_df) > 0 else 0

        def extract_frequent_words(texts: pd.Series, min_frequency=5):
            if texts.empty or texts.isnull().all(): return []
            word_freq = Counter(texts.str.cat(sep=' ').split())
            frequent_words = [(word, count) for word, count in word_freq.items() if count >= min_frequency]
            return sorted(frequent_words, key=lambda x: x[1], reverse=True)

        word_frequencies = {
            'high_rating_words': extract_frequent_words(high_ratings_df.get('Processed_Text'), min_frequency=5),
            'low_rating_words': extract_frequent_words(low_ratings_df.get('Processed_Text'), min_frequency=5)
        }

        print("✅ 产品优缺点报告生成完毕。")
        return {
            'feature_sentiment_stats': feature_sentiment_stats,
            'rating_group_mention_rates': rating_group_mention_rates,
            'word_frequencies': word_frequencies
        }

    def run_analysis(self):
        """按顺序执行完整的核心分析流程。"""
        if self._load_and_clean_data():
            self.analyze_sentiment()
            self.extract_keywords()
            self.categorize_products()
            self._precompute_feature_sentiments()
            self.full_df = self.df.copy() # 创建一个完整的“数据快照”
            print("\n✅ 核心分析流程全部完成！")
            return self.df
        else:
            print("\n❌ 分析因错误而终止。")
            return None



    def save_results(self):
        """将处理后的DataFrame保存到CSV文件。"""
        if self.df is not None:
            output_path = self.config['output_filepath']
            print(f"\n正在将结果保存至 '{output_path}'...")
            self.df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print("结果保存成功。")

    def _analyze_segment_details(self, segment_df: pd.DataFrame, segment_name: str) -> Dict:
        if segment_df.empty: return {"error": "数据不足"}
        return {
            "review_count": len(segment_df),
            "percentage_of_total": f"{(len(segment_df) / len(self.df)) * 100:.2f}%",
            "user_profile": {"role_distribution": segment_df['User_Role'].value_counts().to_dict(), "usage_distribution": segment_df['Usage'].value_counts().to_dict()},
            "product_preference": segment_df['Product_Category'].value_counts().to_dict(),
            "correlated_feature_mentions": {k.replace('feature_', ''): v for k, v in (segment_df[[f'feature_{f}' for f in self.config['feature_keywords'] if f not in segment_name]].mean() * 100).round(2).sort_values(ascending=False).to_dict().items()}
        }



    def deep_dive_feature_analysis(self, feature_name: str, sentiment: str) -> Dict:
        """【V5.2 升级版】: 返回分析结果字典，而不是打印。"""
        sentiment_map, sentiment_text = ({'positive': 1, 'negative': -1}, "正面评价" if sentiment == 'positive' else "负面评价")

        report = {
            "type": "feature_drill_down",
            "title": f"关于【{feature_name}】的【{sentiment_text}】",
            "data": {}
        }

        segment_df = self.df[(self.df[f'feature_{feature_name}'] == 1) & (self.df[f'sentiment_{feature_name}'] == sentiment_map[sentiment])].copy()
        segment_size = len(segment_df)

        if segment_size < 3:
            report["insufficient_data"] = True
            return report

        macro_report = self._analyze_segment_details(segment_df, feature_name)

        report["data"]["summary"] = f"共找到 {macro_report['review_count']} 条相关评论, 占总评论数的 {macro_report['percentage_of_total']}。"
        roles_dist = macro_report['user_profile']['role_distribution']
        usages_dist = macro_report['user_profile']['usage_distribution']
        gender_dist = segment_df['Gender'].value_counts().to_dict()
        age_dist = segment_df['Age_Group'].value_counts().to_dict()
        report["data"]["user_profile"] = {
            "roles": {role: f"{count}次 ({(count / segment_size) * 100:.1f}%)" for role, count in sorted(roles_dist.items(), key=lambda item: item[1], reverse=True)[:5]},
            "usages": {usage: f"{count}次 ({(count / segment_size) * 100:.1f}%)" for usage, count in sorted(usages_dist.items(), key=lambda item: item[1], reverse=True)[:5]},
            "gender_distribution": gender_dist,
            "age_distribution": age_dist
        }
        report["data"]["product_preferences"] = {product: f"{count}次 ({(count / segment_size) * 100:.1f}%)" for product, count in sorted(macro_report['product_preference'].items(), key=lambda item: item[1], reverse=True)[:3]}

        sub_topic_analysis = {}
        feature_sub_topics = self.config['feature_keywords'].get(feature_name, {})
        for sub_topic, keywords in feature_sub_topics.items():
          if sentiment == 'negative' and sub_topic.startswith('正面'):
            continue
          if sentiment == 'positive' and sub_topic.startswith('负面'):
            continue
          if not keywords: continue
          pattern = r'\b(' + '|'.join([re.escape(kw) for kw in keywords]) + r')\b'
          count = segment_df['Processed_Text'].str.contains(pattern, regex=True, na=False).sum()
          if count > 0:
            sub_topic_analysis[sub_topic] = f"{count} 次 ({(count / segment_size) * 100:.1f}%)"
        report["data"]["main_reasons"] = sub_topic_analysis


        related_needs = {}
        for other_feature, mention_rate in sorted(macro_report['correlated_feature_mentions'].items(), key=lambda item: item[1], reverse=True)[:2]:
            sub_needs = {}
            other_feature_sub_topics = self.config['feature_keywords'].get(other_feature, {})
            for sub_topic, keywords in other_feature_sub_topics.items():
                pattern = r'\b(' + '|'.join([re.escape(kw) for kw in keywords]) + r')\b'
                count = segment_df['Processed_Text'].str.contains(pattern, regex=True).sum()
                if count > 0:
                    sub_needs[sub_topic] = f"{count} 次 ({(count / segment_size) * 100:.1f}%)"
            if sub_needs:
                related_needs[other_feature] = {"mention_rate": f"{mention_rate}%", "details": sub_needs}
        report["data"]["related_needs"] = related_needs

        return report




    def analyze_top_praises(self, segment_df: pd.DataFrame, top_n: int = 5) -> Dict:
        """
        【新增功能】
        分析并返回被赞扬次数最多的N个具体原因。
        此函数逻辑与 analyze_top_complaints 完全对应，仅分析高分评论。
        """
        if segment_df.empty:
            return {}

        positive_reviews_df = segment_df[segment_df[self.config['rating_column']] >= 4].copy()
        total_positive_reviews = len(positive_reviews_df)

        if total_positive_reviews == 0:
            return {}

        praise_counts = {}

        for feature, sub_topics in self.config.get('feature_keywords', {}).items():
            for sub_topic, keywords in sub_topics.items():
                if sub_topic.startswith('负面'):
                    continue
                if not keywords: continue
                pattern = r'\b(' + '|'.join([re.escape(kw) for kw in keywords]) + r')\b'
                count = int(positive_reviews_df['Processed_Text'].str.contains(pattern, regex=True, na=False).sum())

                if count > 0:
                    praise_key = f"{feature} » {sub_topic}"
                    praise_counts[praise_key] = count

        if not praise_counts:
            return {}

        sorted_praises = sorted(praise_counts.items(), key=lambda item: item[1], reverse=True)

        formatted_praises = {}
        for key, count in sorted_praises[:top_n]:
            percentage = (count / total_positive_reviews) * 100
            formatted_praises[key] = f"{count} 次 ({percentage:.1f}%)"

        return formatted_praises


    def analyze_top_complaints(self, segment_df: pd.DataFrame, top_n: int = 5) -> Dict:
        """
        【V5.10 最终格式统一版】
        直接分析并返回被抱怨次数最多的N个具体原因。
        输出的字符串格式与“最满意点”完全一致，包含次数和比例。
        """
        if segment_df.empty:
            return {}

        negative_reviews_df = segment_df[segment_df[self.config['rating_column']] <= 3].copy()
        total_negative_reviews = len(negative_reviews_df)

        if total_negative_reviews == 0:
            return {}

        complaint_counts = {}

        for feature, sub_topics in self.config.get('feature_keywords', {}).items():
            for sub_topic, keywords in sub_topics.items():
                if sub_topic.startswith('正面'):
                    continue
                if not keywords: continue
                pattern = r'\b(' + '|'.join([re.escape(kw) for kw in keywords]) + r')\b'
                count = int(negative_reviews_df['Processed_Text'].str.contains(pattern, regex=True, na=False).sum())

                if count > 0:
                    complaint_key = f"{feature} » {sub_topic}"
                    complaint_counts[complaint_key] = count

        if not complaint_counts:
            return {}

        sorted_complaints = sorted(complaint_counts.items(), key=lambda item: item[1], reverse=True)

        formatted_complaints = {}
        for key, count in sorted_complaints[:top_n]:
            percentage = (count / total_negative_reviews) * 100
            formatted_complaints[key] = f"{count} 次 ({percentage:.1f}%)"

        return formatted_complaints



# ▼▼▼▼▼ “特征提升度”分析 (Feature Lift Analysis) ▼▼▼▼▼
    def _calculate_feature_lift(self, segment_df: pd.DataFrame) -> Dict:
        """计算指定用户群体中，各个特征相对于全体用户的“提升度”。"""

        lift_scores = {}
        all_feature_cols = [col for col in self.df.columns if col.startswith('feature_')]

        # 1. 计算每个特征在【全体用户】中的平均提及率
        overall_mention_rates = self.full_df[all_feature_cols].mean()

        # 2. 计算每个特征在【当前群体】中的平均提及率
        segment_mention_rates = segment_df[all_feature_cols].mean()

        for feature_col in all_feature_cols:
            overall_rate = overall_mention_rates[feature_col]
            segment_rate = segment_mention_rates[feature_col]

            # 3. 计算提升度 (群体提及率 / 总体提及率)
            #    为了避免除以零的错误，如果总体提及率为0，则提升度为1（无提升）
            if overall_rate > 0:
                lift = segment_rate / overall_rate
                lift_scores[feature_col.replace('feature_', '')] = f"{lift:.2f}x"
            else:
                lift_scores[feature_col.replace('feature_', '')] = "1.00x" # N/A or 1.00x

        # 返回按提升度从高到低排序的结果
        return dict(sorted(lift_scores.items(), key=lambda item: float(item[1][:-1]), reverse=True))



    def deep_dive_user_segment_analysis(self, attribute_column: str, segment_value: str) -> Dict:
        """
        【V6.1 最终对称分析版】:
        - “最满意点”和“最不满意点”采用完全相同的分析逻辑，都生成全局Top 10的综合排行榜。
        - 确保输出的数据结构完全对称，便于网页端统一处理。
        - 这是最理想、最强大的分析模式。
        """
        report = {
            "type": "user_drill_down",
            "title": f"用户群体深度诊断: 【{segment_value}】",
            "data": {}
        }

        segment_df = self.df[self.df[attribute_column] == segment_value].copy()
        segment_size = len(segment_df)
        if segment_size < 3:
            report["insufficient_data"] = True
            return report

        # --- 模块1: 群体概览 (Overview) ---
        report["data"]["summary"] = f"共找到 {segment_size} 条相关评论, 占总评论数的 {(segment_size / len(self.df)) * 100:.2f}%."

        motivations = {}
        if 'Motivation' in segment_df.columns:
            motivation_dist = segment_df['Motivation'].value_counts()
            for motivation, count in motivation_dist.head(3).items():
                motivations[motivation] = f"{count}次 ({(count / segment_size) * 100:.1f}%)"

        products = {}
        product_pref = segment_df['Product_Category'].value_counts()
        for product, count in product_pref.head(3).items():
            products[product] = f"{count}次 ({(count / segment_size) * 100:.1f}%)"


        # 计算并添加该用户群体的性别构成
        gender_dist = segment_df['Gender'].value_counts().to_dict()

        # 计算并添加该用户群体的年龄段构成
        age_dist = segment_df['Age_Group'].value_counts().to_dict()

        report['data']['overview'] = {"motivations": motivations,
                        "products": products,
                        "gender_distribution": gender_dist,
                        "age_distribution": age_dist
        }

        # --- 模块2: 核心需求 (Core Needs) ---
        feature_sentiments = {}
        all_features = self.config.get('feature_keywords', {}).keys()
        for feature in all_features:
            total_mentions = segment_df[f'feature_{feature}'].sum()
            if total_mentions == 0: continue
            positive_mentions = (segment_df[f'sentiment_{feature}'] == 1).sum()
            negative_mentions = (segment_df[f'sentiment_{feature}'] == -1).sum()
            feature_sentiments[feature] = {
                'mention_rate': (total_mentions / segment_size) * 100,
                'positive_ratio': (positive_mentions / total_mentions) * 100 if total_mentions > 0 else 0,
                'negative_ratio': (negative_mentions / total_mentions) * 100 if total_mentions > 0 else 0,
            }

        sorted_features = sorted(feature_sentiments.items(), key=lambda item: item[1]['mention_rate'], reverse=True)
        report['data']['core_needs'] = {f: f"关注度 {d['mention_rate']:.1f}% (好评率: {d['positive_ratio']:.1f}%, 差评率: {d['negative_ratio']:.1f}%)" for f, d in sorted_features[:5]}

        # --- 模块3: 关联需求 (Correlated Needs) ---
        report['data']['correlated_needs'] = {}
        all_feature_cols = [f'feature_{f}' for f in self.config.get('feature_keywords', {}).keys()]
        existing_feature_cols = [col for col in all_feature_cols if col in segment_df.columns]
        if len(existing_feature_cols) > 1:
            corr_matrix = segment_df[existing_feature_cols].corr()
            corr_pairs = corr_matrix.unstack()
            sorted_pairs = corr_pairs.sort_values(ascending=False)
            top_pairs = sorted_pairs[sorted_pairs.index.get_level_values(0) != sorted_pairs.index.get_level_values(1)]
            unique_pairs = {}
            for (f1, f2), corr in top_pairs.items():
                pair_key = tuple(sorted((f1.replace('feature_', ''), f2.replace('feature_', ''))))
                if pair_key not in unique_pairs:
                    if corr > 0.05:
                        unique_pairs[pair_key] = corr
            top_correlated_features = sorted(unique_pairs.items(), key=lambda item: item[1], reverse=True)[:3]
            correlated_needs_result = {f"'{pair[0]}' 与 '{pair[1]}'": f"关联度: {corr:.2f}" for pair, corr in top_correlated_features}
            report['data']['correlated_needs'] = correlated_needs_result

        # --- 模块4: 深度原因剖析 (Deep Dive Reasons) ---
        report['data']['deep_dive_reasons'] = {}

        # 4.1【最满意点】直接调用 analyze_top_praises，生成全局排行榜
        top_praises_data = self.analyze_top_praises(segment_df, top_n=10)
        if top_praises_data:
            report['data']['deep_dive_reasons']['最满意点: 【综合优点 Top 10】'] = top_praises_data

        # 4.2【最不满意点】的逻辑保持不变，它已经是全局排行榜
        top_complaints_data = self.analyze_top_complaints(segment_df, top_n=10)
        if top_complaints_data:
            report['data']['deep_dive_reasons']['最不满意点: 【综合痛点 Top 10】'] = top_complaints_data

        lift_analysis_results = self._calculate_feature_lift(segment_df)
        report['data']['signature_needs_lift'] = lift_analysis_results

        return report




    def run_comprehensive_user_diagnostics(self) -> List[Dict]:
        """【V5.2 升级版】: 收集并返回所有用户群体的诊断报告。"""
        print("\n\n" + "#"*70)
        print("####  正在执行【基于用户属性的自动化深度诊断】...  ####")
        print("#"*70)

        all_reports = []
        attributes_to_analyze = ['User_Role']
        for column in attributes_to_analyze:
            segments = self.df[column].unique()
            for segment in segments:
                report = self.deep_dive_user_segment_analysis(attribute_column=column, segment_value=segment)
                all_reports.append(report)
        return all_reports

    def run_comprehensive_feature_diagnostics(self) -> List[Dict]:
        """【V5.2 升级版】: 收集并返回所有特征的诊断报告。"""
        print("\n\n" + "#"*70 + "\n####  正在执行【全特征自动化深度诊断分析】...  ####\n" + "#"*70)
        all_reports = []
        for feature in self.config.get('feature_keywords', {}).keys():
            report_pos = self.deep_dive_feature_analysis(feature, sentiment='positive')
            all_reports.append(report_pos)
            report_neg = self.deep_dive_feature_analysis(feature, sentiment='negative')
            all_reports.append(report_neg)
        return all_reports


    def export_to_html(self, dashboard_data: Dict):
        """
        将分析数据注入HTML模板，并生成最终的报告网页。
        您应该将自己的完整HTML/CSS/JS代码替换掉下面的占位符。
        """
        template_str = """
        <!DOCTYPE html>
        <!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>产品评论数据洞察报告 (V5.5 - 修正完整版)</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js" defer></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.0.0" defer></script>
    <script src="https://cdn.jsdelivr.net/npm/d3@7" defer></script>
    <script src="https://cdn.jsdelivr.net/npm/d3-cloud@1/build/d3.layout.cloud.js" defer></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #7F80BA; --secondary-color: #BDAECD; --accent-color: #F7D691;
            --light-accent: #F7ECD9; --text-color: #444; --light-text: #777;
            --background: #F8F9FA; --card-bg: #FFF; --border-color: #e4e4e4;
            --success: #63BF84; --danger: #E57A77;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: var(--background); color: var(--text-color); line-height: 1.6; }
        .container { max-width: 1600px; margin: 0 auto; padding: 20px; }
        header { background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)); color: white; padding: 30px 0; text-align: center; margin-bottom: 30px; border-radius: 0 0 10px 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .dashboard-section { background-color: var(--card-bg); border-radius: 8px; margin-bottom: 30px; padding: 25px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .section-header { border-bottom: 2px solid var(--secondary-color); padding-bottom: 15px; margin-bottom: 20px; color: var(--primary-color); display: flex; justify-content: space-between; align-items: center; }
        h1, h2 { margin: 0; } h1 { font-size: 2.2rem; } h2 { font-size: 1.8rem; }
        .flex-container { display: flex; flex-wrap: wrap; gap: 20px; align-items: stretch; }
        .flex-item { flex: 1; min-width: 300px; }
        .stat-card { background: white; border-radius: 8px; padding: 20px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05); height: 100%; display: flex; flex-direction: column; justify-content: center;}
        .stat-card h3 { margin: 0 0 15px; color: var(--light-text); font-weight: 500; font-size: 1rem; }
        .stat-card .value { font-size: 2.5rem; font-weight: bold; color: var(--primary-color); }
        .chart-container { position: relative; width: 100%; min-height: 350px; }
        .word-cloud-container { width: 100%; height: 350px; }
        .no-data-placeholder { display: flex; align-items: center; justify-content: center; width: 100%; height: 100%; color: var(--light-text); font-style: italic; background: #fafafa; border-radius: 5px; min-height: 200px; }
        .master-detail-container .report-nav { max-height: 800px; overflow-y: auto; border-right: 1px solid var(--border-color); }
        /* Style for the select dropdown */
        .master-detail-container .form-select { cursor: pointer; }
        .report-detail-view { padding-left: 25px; }
        .report-detail-view .detail-chart-container { min-height: 280px; margin-bottom: 1rem; }
        .report-detail-view h4 { color: var(--primary-color); margin-bottom: 0.5rem; }
        .report-detail-view .summary-text { color: var(--light-text); margin-bottom: 2rem; }
        .report-detail-view h5 { color: var(--secondary-color); border-bottom: 1px solid #eee; padding-bottom: 8px; margin-top: 2rem; margin-bottom: 1rem; font-size: 1.2rem; font-weight: 600;}
        .sub-section-card { background-color: #fcfcfc; border: 1px solid #eee; border-left: 4px solid var(--accent-color); padding: 15px; border-radius: 4px; margin-bottom: 1rem;}
        .correlation-item .progress { height: 20px; font-size: 0.85rem; }
        .correlation-item .progress-bar { font-weight: bold; background-color: var(--primary-color); }
        .correlation-item .badge { font-size: 0.9em; }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>产品评论数据洞察报告</h1>
            <p>基于 <span id="totalReviews">...</span> 条评论数据的分析 (时间维度版)</p>
        </div>
    </header>
    
    <div class="container">
        <div class="dashboard-section">
             <div class="section-header"><h2>总体评价概览</h2></div>
             <div class="flex-container">
                 <div class="flex-item"><div class="stat-card"><h3>总评论数</h3><div class="value" id="totalReviewsCard">...</div></div></div>
                 <div class="flex-item"><div class="stat-card"><h3>平均评分</h3><div class="value" id="avgRatingCard">...</div></div></div>
                 <div class="flex-item"><div class="stat-card"><h3>好评率 (4-5星)</h3><div class="value" id="positiveRateCard">...</div></div></div>
             </div>
             <div class="chart-container" style="margin-top:20px;"><canvas id="ratingDistributionChart"></canvas></div>
        </div>
         <div class="dashboard-section">
             <div class="section-header"><h2>用户画像与行为分析</h2></div>
             <div class="flex-container">
                 <div class="flex-item"><div class="chart-container"><canvas id="userRolesChart"></canvas></div></div>
                 <div class="flex-item"><div class="chart-container"><canvas id="usageAnalysisChart"></canvas></div></div>
                 <div class="flex-item"><div class="chart-container"><canvas id="purchaseMotivationChart"></canvas></div></div>
             </div>
        </div>
         <div class="dashboard-section">
             <div class="section-header"><h2>核心产品特征分析</h2></div>
             <div class="chart-container" style="height: 400px; margin-bottom: 40px;"><canvas id="featureMentionRateChart"></canvas></div>
        </div>
         <div class="dashboard-section">
             <div class="section-header"><h2>口碑词云对比</h2></div>
             <div class="flex-container">
                 <div class="flex-item">
                     <h3 style="text-align:center; margin-bottom: 15px; color: var(--success);">好评关键词 (4-5星)</h3>
                     <div id="highRatingWordCloud" class="word-cloud-container"></div>
                 </div>
                 <div class="flex-item">
                     <h3 style="text-align:center; margin-bottom: 15px; color: var(--danger);">差评关键词 (1-3星)</h3>
                     <div id="lowRatingWordCloud" class="word-cloud-container"></div>
                 </div>
             </div>
        </div>
        
        <div class="dashboard-section">
            <div class="section-header">
                <h2>深度下钻分析报告</h2>
                <div class="col-md-3">
                    <select class="form-select" id="timePeriodSelector"></select>
                </div>
            </div>
            <p class="text-muted mb-4">请首先选择时间范围，然后从左侧选择要查看的分析报告，右侧将呈现详细洞察。</p>
            <div class="row master-detail-container">
                <div class="col-lg-4 col-md-5 report-nav">
                    <label for="drillDownNav" class="form-label fw-bold">选择具体分析报告:</label>
                    <select class="form-select" id="drillDownNav"></select>
                </div>
                <div class="col-lg-8 col-md-7 report-detail-view">
                    <div id="drillDownDetail">
                        <div class="no-data-placeholder h-100">请选择时间段并选择一个报告查看详情</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <footer class="mt-5 py-4" style="background-color: var(--primary-color); color: white; text-align:center; border-radius: 10px 10px 0 0;">
        <p class="m-0">© <span id="footerYear"></span> 产品洞察报告. 生成时间: <span id="generationTime"></span></p>
    </footer>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js" defer></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const rawData = __DATA_PLACEHOLDER__;

            const App = {
                data: {},
                detailCharts: [],
                init() {
                    Chart.defaults.scale.grid.display = false;
                    this.data = { ...rawData };
                    setTimeout(() => {
                        if (typeof Chart === 'undefined' || typeof d3 === 'undefined' || !d3.layout?.cloud) {
                            document.body.innerHTML = `<h1 style="text-align:center; padding:50px;">错误：核心JS库加载失败。</h1>`; return;
                        }
                        Chart.register(ChartDataLabels);
                        this.render();
                    }, 150);
                },
                render() {
                    this.renderKPIs();
                    this.renderMainCharts();
                    this.renderWordClouds();
                    this.renderDrillDownReports();
                },
                renderKPIs() {
                    document.getElementById('totalReviews').textContent = this.data.totalReviews;
                    document.getElementById('totalReviewsCard').textContent = this.data.totalReviews;
                    document.getElementById('avgRatingCard').textContent = this.data.avgRating;
                    document.getElementById('positiveRateCard').textContent = this.data.positiveRate;
                    document.getElementById('footerYear').textContent = new Date().getFullYear();
                    document.getElementById('generationTime').textContent = new Date().toLocaleString('zh-CN');
                },
                renderMainCharts() {
                    const chartColors = ['#7F80BA', '#BDAECD', '#F7D691', '#85C1E9', '#73C6B6', '#F0B27A', '#C39BD3'];
                    const pieOptions = (title) => ({ responsive: true, maintainAspectRatio: false, plugins: { title: { display: true, text: title, font: {size: 16}}, legend: { position: 'bottom' }, datalabels: { formatter: (val, ctx) => ((val / ctx.chart.data.datasets[0].data.reduce((a, b) => a + b, 0)) * 100 > 5) ? `${( (val / ctx.chart.data.datasets[0].data.reduce((a, b) => a + b, 0)) * 100).toFixed(0)}%` : '', color: '#fff', font: { weight: 'bold' } } } });
                    this.createChart('userRolesChart', 'doughnut', { labels: this.data.userRoles?.labels, datasets: [{ data: this.data.userRoles?.data, backgroundColor: chartColors }] }, pieOptions('用户角色分布'));
                    this.createChart('usageAnalysisChart', 'doughnut', { labels: this.data.usageAnalysis?.labels, datasets: [{ data: this.data.usageAnalysis?.data, backgroundColor: chartColors }] }, pieOptions('主要用途分析'));
                    this.createChart('purchaseMotivationChart', 'doughnut', { labels: this.data.purchaseMotivation?.labels, datasets: [{ data: this.data.purchaseMotivation?.data, backgroundColor: chartColors }] }, pieOptions('购买动机分析'));
                    this.createChart('ratingDistributionChart', 'bar', { labels: this.data.ratingDistribution?.labels, datasets: [{ label: '评论数', data: this.data.ratingDistribution?.data, backgroundColor: ['#E57A77', '#F7C773', '#F7D691', '#94BF8E', '#63BF84'] }] }, { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, datalabels: { color: '#fff', anchor: 'center', align: 'center', font: { weight: 'bold' }, formatter: (v, ctx) => `${v}\n(${(v / ctx.chart.data.datasets[0].data.reduce((a, b) => a + b, 0) * 100).toFixed(1)}%)` } } });
                    if (this.data.featureSentimentStats) {
                        const labels = Object.keys(this.data.featureSentimentStats).sort((a,b) => this.data.featureSentimentStats[b].total_mentions - this.data.featureSentimentStats[a].total_mentions);
                        this.createChart('featureSentimentChart', 'bar', { labels, datasets: [ { label: '正面率', data: labels.map(l => this.data.featureSentimentStats[l].positive_ratio), backgroundColor: 'rgba(99, 191, 132, 0.7)' }, { label: '负面率', data: labels.map(l => this.data.featureSentimentStats[l].negative_ratio), backgroundColor: 'rgba(229, 122, 119, 0.7)' } ] }, { indexAxis: 'y', responsive: true, maintainAspectRatio: false, scales: { x: { stacked: true, ticks: { callback: v => v + '%' } ,max: 100}, y: { stacked: true } }, plugins: { title: {display: true, text: '各特征情感倾向 (按总提及数排序)'}, legend: { position: 'bottom' }, datalabels: { color: '#fff', font: { weight: 'bold' }, formatter: (v) => v > 5 ? v.toFixed(0) + '%' : '' } } });
                    }
                    if (this.data.featureMentionRates) {
                        const labels = Object.keys(this.data.featureMentionRates.high_ratings).sort((a,b) => this.data.featureMentionRates.high_ratings[b] - this.data.featureMentionRates.high_ratings[a]);
                        this.createChart('featureMentionRateChart', 'bar', { labels, datasets: [ { label: '高分提及率', data: labels.map(l => this.data.featureMentionRates.high_ratings[l]), backgroundColor: 'rgba(99, 191, 132, 0.7)' }, { label: '低分提及率', data: labels.map(l => this.data.featureMentionRates.low_ratings[l]), backgroundColor: 'rgba(229, 122, 119, 0.7)' } ] }, { responsive: true, maintainAspectRatio: false, scales: { y: { ticks: { callback: v => v.toFixed(1) + '%' } } }, plugins: { title: {display: true, text: '高/低分评价下的特征提及率 (按高分提及率排序)'}, legend: { position: 'bottom' }, datalabels: { anchor: 'end', align: 'top', formatter: (v) => v.toFixed(1) + '%', font: { weight: 'bold' }, color: '#444' } } });
                    }
                },
                createChart(canvasId, type, data, options) {
                    const canvas = document.getElementById(canvasId); if (!canvas) return;
                    const container = canvas.parentElement;
                    if (!data || !data.labels || !data.datasets.every(ds => ds.data && ds.data.length > 0 && ds.data.some(d => d > 0))) {
                        if(container) container.innerHTML = `<div class="no-data-placeholder">无可用图表数据</div>`; return;
                    }
                    new Chart(canvas.getContext('2d'), { type, data, options });
                },
                renderWordClouds() {
                    this.createWordCloud('highRatingWordCloud', this.data.highRatingWordCloudData, ['#7F80BA', '#63BF84', '#6AAFE6']);
                    this.createWordCloud('lowRatingWordCloud', this.data.lowRatingWordCloudData, ['#BDAECD', '#E57A77', '#F5C06A']);
                },
                createWordCloud(containerId, words, colorRange) {
                    const container = document.getElementById(containerId);
                    if (!container || !words || words.length === 0) {
                        if(container) container.innerHTML = `<div class="no-data-placeholder">无可用词云数据</div>`; return;
                    }
                    container.innerHTML = '';
                    const color = d3.scaleOrdinal().range(colorRange);
                    const layout = d3.layout.cloud().size([container.clientWidth, 350]).words(words.map(d => ({text: d.text, size: d.size}))).padding(5).rotate(() => (Math.random() > 0.7) ? 90 : 0).fontSize(d => d.size * 1.5 + 12).on("end", draw);
                    layout.start();
                    function draw(drawnWords) { d3.select(container).append("svg").attr("width", layout.size()[0]).attr("height", layout.size()[1]).append("g").attr("transform", `translate(${layout.size()[0] / 2},${layout.size()[1] / 2})`).selectAll("text").data(drawnWords).enter().append("text").style("font-size", d => `${d.size}px`).style("fill", d => color(d.text)).attr("text-anchor", "middle").attr("transform", d => `translate(${d.x},${d.y})rotate(${d.rotate})`).text(d => d.text); }
                },

                // ======================= JS 修改 1: 更新事件监听逻辑 =======================
                renderDrillDownReports() {
                    const timeSelector = document.getElementById('timePeriodSelector');
                    const navSelect = document.getElementById('drillDownNav'); // 获取 select 元素

                    if (!timeSelector || !this.data.drillDownTimePeriods || Object.keys(this.data.drillDownTimePeriods).length === 0) {
                        document.querySelector('.master-detail-container').innerHTML = `<div class="no-data-placeholder">无下钻分析数据</div>`;
                        return;
                    }
                    
                    let timeOptionsHtml = '';
                    Object.entries(this.data.drillDownTimePeriods).forEach(([key, label]) => {
                        if (this.data.drillDownReports[key] && this.data.drillDownReports[key].length > 0) {
                            timeOptionsHtml += `<option value="${key}">${label}</option>`;
                        } else {
                            timeOptionsHtml += `<option value="${key}" disabled>${label} (无足够数据)</option>`;
                        }
                    });
                    timeSelector.innerHTML = timeOptionsHtml;
                    
                    // 当“时间段”变化时，重新填充“具体报告”下拉菜单
                    timeSelector.addEventListener('change', (event) => {
                        this.renderDrillDownNavForPeriod(event.target.value);
                    });
                    
                    // 监听下拉菜单的 "change" 事件，而不是 "click"
                    navSelect.addEventListener('change', (event) => {
                        const period = timeSelector.value; // 从时间选择器获取 period
                        const index = parseInt(event.target.value, 10); // 从下拉菜单获取 index
                        this.displayReportDetail(period, index);
                    });

                    // 初始化时，为第一个有效的时间段填充报告列表
                    this.renderDrillDownNavForPeriod(timeSelector.value);
                },

                // ======================= JS 修改 2: 更新列表填充逻辑 =======================
                renderDrillDownNavForPeriod(periodKey) {
                    const navSelect = document.getElementById('drillDownNav'); // 获取 select 元素
                    const detailContainer = document.getElementById('drillDownDetail');
                    const reportsForPeriod = this.data.drillDownReports[periodKey] || [];
                    
                    navSelect.innerHTML = ''; // 清空旧的 options
                    detailContainer.innerHTML = `<div class="no-data-placeholder h-100">请从下拉菜单中选择一个报告查看详情</div>`;

                    if (reportsForPeriod.length === 0) {
                        navSelect.innerHTML = `<option disabled selected>当前时间段无有效分析报告</option>`;
                        return;
                    }

                    // 生成 <option> 而不是 <a>
                    let navHtml = '';
                    reportsForPeriod.forEach((report, index) => {
                        if (report.insufficient_data) return;
                        navHtml += `<option value="${index}">${report.title}</option>`;
                    });
                    navSelect.innerHTML = navHtml;
                    
                    // 自动触发一次 change 事件来显示第一个报告的详情
                    if (navSelect.options.length > 0) {
                        navSelect.dispatchEvent(new Event('change'));
                    }
                },
                
                displayReportDetail(periodKey, index) {
                    this.detailCharts.forEach(chart => chart.destroy());
                    this.detailCharts = [];
                    
                    const report = this.data.drillDownReports[periodKey][index];
                    const container = document.getElementById('drillDownDetail');
                    if (!report || !container) return;
                    
                    const summaryText = report.data.summary || '';
                    let detailHtml = `<h4>${report.title}</h4><p class="summary-text">${summaryText}</p>`;
                    
                    if (report.type === 'feature_drill_down') {
                        detailHtml += `
                            <h5>用户画像</h5>
                            <div class="row">
                                <div class="col-md-6 detail-chart-container"><canvas id="dd-roles-chart"></canvas></div>
                                <div class="col-md-6 detail-chart-container"><canvas id="dd-usages-chart"></canvas></div>
                                <div class="col-md-6 detail-chart-container"><canvas id="dd-gender-chart"></canvas></div>
                                <div class="col-md-6 detail-chart-container"><canvas id="dd-age-chart"></canvas></div>
                            </div>
                            <h5>产品偏好</h5>
                            <div class="detail-chart-container" style="height:300px;"><canvas id="dd-product-prefs-chart"></canvas></div>
                            <h5>主要原因剖析</h5>
                            <div class="detail-chart-container" style="height:300px;"><canvas id="dd-main-reasons-chart"></canvas></div>
                            <div id="dd-related-needs-container"></div>
                        `;
                    } else if (report.type === 'user_drill_down') {
                        detailHtml += `
                            <h5>群体概览</h5>
                            <div class="row">
                                <div class="col-md-6 detail-chart-container"><canvas id="dd-motivations-chart"></canvas></div>
                                <div class="col-md-6 detail-chart-container"><canvas id="dd-products-chart"></canvas></div>
                                <div class="col-md-6 detail-chart-container"><canvas id="dd-gender-dist-chart"></canvas></div>
                                <div class="col-md-6 detail-chart-container"><canvas id="dd-age-dist-chart"></canvas></div>
                            </div>
                            <h5>核心需求洞察</h5>
                            <div class="detail-chart-container" style="height:300px;"><canvas id="dd-core-needs-chart"></canvas></div>

                            <h5>标志性需求 (与总体对比的提升度)</h5>
                            <div class="detail-chart-container" style="height:300px;"><canvas id="dd-signature-needs-chart"></canvas></div>
                            <div id="dd-correlated-needs-container"></div>
                            <div id="dd-deep-dive-reasons-container"></div>
                        `;
                    }

                    container.innerHTML = detailHtml;
                    setTimeout(() => { this.renderDetailContent(report); }, 50);
                },

                renderDetailContent(report) {
                    const horizontalBarOptions = (title) => ({ indexAxis: 'y', responsive: true, maintainAspectRatio: false, scales: { x: { ticks: { callback: value => value + '%' } } }, plugins: { title: { display: true, text: title }, legend: { display: false }, tooltip: { callbacks: { label: (ctx) => `${ctx.raw.toFixed(1)}%` } }, datalabels: { anchor: 'end', align: 'end', color: '#555', formatter: (val) => `${val.toFixed(1)}%` } } });
                    const doughnutOptions = (title) => ({ responsive: true, maintainAspectRatio: false, plugins: { title: { display: true, text: title }, legend: { position: 'right' }, tooltip: { callbacks: { label: (ctx) => ` ${ctx.label}: ${ctx.formattedValue} (${(ctx.raw / ctx.chart.getDatasetMeta(0).total * 100).toFixed(1)}%)` } }, datalabels: { formatter: (val, ctx) => `${(val / ctx.chart.getDatasetMeta(0).total * 100).toFixed(1)}%`, color: '#fff', font: {weight: 'bold'} } } });
                    const chartColors1 = ['#7F80BA', '#BDAECD', '#F7D691', '#85C1E9', '#73C6B6'];
                    const chartColors2 = ['#F0B27A', '#C39BD3', '#76D7C4', '#F7DC6F', '#E59866'];
                    const parseChartData = (dataDict, valueKey = null, dataType = 'string') => {
                        if (!dataDict || Object.keys(dataDict).length === 0) return { labels: [], values: [] };
                        const labels = Object.keys(dataDict);
                        const values = labels.map(label => {
                            const item = dataDict[label];
                            if (valueKey && typeof item === 'object') {
                                return (dataType === 'count' && item.hasOwnProperty('count')) ? item.count : (item[valueKey] || 0);
                            }
                            if (typeof item === 'string') {
                                let match;
                                switch (dataType) {
                                      case 'count':
                                             match = item.match(/^(\d+)/); // 匹配开头的数字 (次数)
                                             break;
                                      case 'percent':
                                             match = item.match(/([\d\.]+)%/); // 匹配百分比
                                             break;
                                      case 'lift': 
                                             match = item.match(/([\d\.]+)x/); // 匹配 'x' (提升度)
                                             break;
                                      default:
                                             match = item.match(/(-?[\d\.]+)/); 
                                }
                                return match ? parseFloat(match[1]) : 0;
                             }
                            return typeof item === 'number' ? item : 0;
                        });
                        return { labels, values };
                    };
                    
                    if (report.type === 'feature_drill_down') {
                        const rolesData = parseChartData(report.data.user_profile.roles, null, 'count');
                        const usagesData = parseChartData(report.data.user_profile.usages, null, 'count');
                        const productPrefsData = parseChartData(report.data.product_preferences, null, 'count');
                        const mainReasonsData = parseChartData(report.data.main_reasons, 'pct', 'percent');
                        const genderData = parseChartData(report.data.user_profile.gender_distribution, null, 'count');
                        const ageData = parseChartData(report.data.user_profile.age_distribution, null, 'count');
                        this.createDetailChart('dd-roles-chart', 'doughnut', { labels: rolesData.labels, datasets: [{ data: rolesData.values, backgroundColor: chartColors1 }] }, doughnutOptions('用户角色分布'));
                        this.createDetailChart('dd-gender-chart', 'pie', { labels: genderData.labels, datasets: [{ data: genderData.values, backgroundColor: chartColors1 }] }, doughnutOptions('性别构成'));
                        this.createDetailChart('dd-age-chart', 'pie', { labels: ageData.labels, datasets: [{ data: ageData.values, backgroundColor: chartColors2 }] }, doughnutOptions('年龄段构成'));
                        this.createDetailChart('dd-usages-chart', 'doughnut', { labels: usagesData.labels, datasets: [{ data: usagesData.values, backgroundColor: chartColors2 }] }, doughnutOptions('主要用途分布'));
                        this.createDetailChart('dd-product-prefs-chart', 'doughnut', { labels: productPrefsData.labels, datasets: [{ data: productPrefsData.values, backgroundColor: chartColors1.slice(2).concat(chartColors2.slice(2)) }] }, doughnutOptions('产品偏好'));
                        this.createDetailChart('dd-main-reasons-chart', 'bar', { labels: mainReasonsData.labels, datasets: [{ data: mainReasonsData.values, backgroundColor: '#7F80BA' }] }, horizontalBarOptions('具体原因提及比例'));
                        const relatedNeedsContainer = document.getElementById('dd-related-needs-container');
                        if (relatedNeedsContainer && report.data.related_needs && Object.keys(report.data.related_needs).length > 0) {
                            let relatedNeedsHtml = '<h5>关联需求分析</h5>';
                            Object.keys(report.data.related_needs).forEach(feature => {
                                const featureId = `related-${feature.replace(/[^a-zA-Z0-9]/g, '')}`;
                                relatedNeedsHtml += `<div class="detail-chart-container" style="height:200px;"><canvas id="${featureId}"></canvas></div>`;
                            });
                            relatedNeedsContainer.innerHTML = relatedNeedsHtml;
                            setTimeout(() => {
                                Object.entries(report.data.related_needs).forEach(([feature, data]) => {
                                    const featureId = `related-${feature.replace(/[^a-zA-Z0-9]/g, '')}`;
                                    const relatedData = parseChartData(data.details, null, 'percent');
                                    this.createDetailChart(featureId, 'bar', { labels: relatedData.labels, datasets: [{ data: relatedData.values, backgroundColor: '#BDAECD' }] }, horizontalBarOptions(`同时关注: ${feature} (${data.mention_rate})`));
                                });
                            }, 50);
                        }
                    } else if (report.type === 'user_drill_down') {
                        const motivationsData = parseChartData(report.data.overview.motivations, null, 'count');
                        const productsData = parseChartData(report.data.overview.products, null, 'count');
                        this.createDetailChart('dd-motivations-chart', 'doughnut', { labels: motivationsData.labels, datasets: [{ data: motivationsData.values, backgroundColor: chartColors1 }] }, doughnutOptions('主要购买动机'));
                        this.createDetailChart('dd-products-chart', 'doughnut', { labels: productsData.labels, datasets: [{ data: productsData.values, backgroundColor: chartColors2 }] }, doughnutOptions('主要购买产品'));
                        
                        const genderData = parseChartData(report.data.overview.gender_distribution, null, 'count');
                        this.createDetailChart('dd-gender-dist-chart', 'pie', { labels: genderData.labels, datasets: [{ data: genderData.values, backgroundColor: chartColors1 }] }, doughnutOptions('性别构成'));
                        const ageData = parseChartData(report.data.overview.age_distribution, null, 'count');
                        this.createDetailChart('dd-age-dist-chart', 'pie', { labels: ageData.labels, datasets: [{ data: ageData.values, backgroundColor: chartColors2 }] }, doughnutOptions('年龄段构成'));

                        const coreNeedsData = parseChartData(report.data.core_needs, null, 'percent');
                        this.createDetailChart('dd-core-needs-chart', 'bar', { labels: coreNeedsData.labels, datasets: [{ data: coreNeedsData.values, backgroundColor: '#7F80BA' }] }, horizontalBarOptions('特征关注度'));
                        const correlatedNeedsContainer = document.getElementById('dd-correlated-needs-container');
                        if (correlatedNeedsContainer) {
                           correlatedNeedsContainer.innerHTML = this.generateCorrelationListHtml('群体需求组合', report.data.correlated_needs);
                        }
                        if (report.data.signature_needs_lift) {
                            const liftChartOptions = (title) => ({
                                   indexAxis: 'y', responsive: true, maintainAspectRatio: false,
                                   scales: { x: { ticks: { callback: value => value.toFixed(1) + 'x' } } }, // X轴标签带 'x'
                                   plugins: {
                                          title: { display: true, text: title },
                                          legend: { display: false },
                                          tooltip: { callbacks: { label: (ctx) => `提升度: ${ctx.raw.toFixed(2)}x` } },
                                          datalabels: {
                                                 anchor: 'end', align: 'end', color: '#555',
                                                 formatter: (val) => `${val.toFixed(1)}x` // 数据标签带 'x'       
                                           }
                                    }
                            });
                           const liftData = parseChartData(report.data.signature_needs_lift, null, 'lift');
                          this.createDetailChart(
                               'dd-signature-needs-chart', 
                               'bar', 
                              { 
                                     labels: liftData.labels, 
                                    datasets: [{ 
                                               data: liftData.values, 
                                               backgroundColor: '#F7D691' // 使用醒目的强调色
                                    }] 
                              }, 
                              liftChartOptions('标志性需求 (提升度 vs. 总体)')
                            );
                         }

                        const deepDiveContainer = document.getElementById('dd-deep-dive-reasons-container');
                        if (deepDiveContainer && report.data.deep_dive_reasons && Object.keys(report.data.deep_dive_reasons).length > 0) {
                            let deepDiveHtml = '';
                            Object.entries(report.data.deep_dive_reasons).forEach(([title, data], index) => {
                                const id = `deep-dive-chart-${index}`;
                                deepDiveHtml += `<div><h5>${title}</h5><div class="detail-chart-container" style="height:250px;"><canvas id="${id}"></canvas></div></div>`;
                            });
                            deepDiveContainer.innerHTML = deepDiveHtml;
                            setTimeout(() => {
                                Object.entries(report.data.deep_dive_reasons).forEach(([title, data], index) => {
                                    const id = `deep-dive-chart-${index}`;
                                    const parsedData = parseChartData(data, 'pct', 'percent');
                                    this.createDetailChart(id, 'bar', { 
                                        labels: parsedData.labels, 
                                        datasets: [{ 
                                            data: parsedData.values, 
                                            backgroundColor: title.includes('不满意') ? '#BDAECD' : '#BDAECD' 
                                        }] 
                                    }, 
                                    horizontalBarOptions(title)
                                  );
                                });
                            }, 50);
                        }
                    }
                },
                createDetailChart(canvasId, type, data, options) {
                    const canvas = document.getElementById(canvasId);
                    if (!canvas) return;
                    if (!data || !data.labels || data.labels.length === 0 || !data.datasets.every(ds => ds.data && ds.data.length > 0 && ds.data.some(d => d > 0))) {
                        canvas.parentElement.innerHTML = `<div class="no-data-placeholder" style="height:100%">${(options && options.plugins && options.plugins.title) ? options.plugins.title.text : ''} 无可用数据</div>`; return;
                    }
                    const chart = new Chart(canvas.getContext('2d'), { type, data, options });
                    this.detailCharts.push(chart);
                },
                generateCorrelationListHtml(title, dataDict) {
                    if (!dataDict || Object.keys(dataDict).length === 0) return '';
                    let listHtml = '';
                    for (const [key, value] of Object.entries(dataDict)) {
                        const features = key.replace(/'/g, '').split(' 与 ');
                        const score = parseFloat(value.match(/[\d\.]+/)[0]);
                        const percentage = (score * 100).toFixed(1);
                        listHtml += `
                            <div class="correlation-item mb-3">
                                <div class="mb-1">
                                    <span class="badge bg-primary">${features[0]}</span>
                                    <span class="fw-bold mx-1">+</span>
                                    <span class="badge bg-secondary">${features[1]}</span>
                                </div>
                                <div class="progress" role="progressbar" aria-valuenow="${percentage}" aria-valuemin="0" aria-valuemax="100">
                                    <div class="progress-bar" style="width: ${percentage}%;">${score.toFixed(2)}</div>
                                </div>
                            </div>
                        `;
                    }
                    return `<div class="sub-section-card"><h5>${title}</h5>${listHtml}</div>`;
                }
            };

            App.init();
        });
    </script>
</body>
</html>
        """
        output_path = self.config.get('report_output_path', 'report.html')
        try:
            print(f"\n正在生成网页报告...")
            # 使用repr()来处理JSON字符串中的特殊字符，确保JS可以解析
            data_json_str = json.dumps(dashboard_data, indent=4, ensure_ascii=False, default=str)
            final_html = template_str.replace('__DATA_PLACEHOLDER__', data_json_str)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_html)
            print(f"✅ 成功！网页报告已生成: '{output_path}'")
        except Exception as e:
            print(f"❌ 生成网页报告时发生错误: {e}")
