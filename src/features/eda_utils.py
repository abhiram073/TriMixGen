import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import os
import logging

logger = logging.getLogger(__name__)

class DatasetAnalyzer:
    def __init__(self, df: pd.DataFrame, dataset_name: str, text_col: str):
        self.df = df
        self.dataset_name = dataset_name
        self.text_col = text_col
        self.output_dir = os.path.join("outputs", "eda", self.dataset_name)
        os.makedirs(self.output_dir, exist_ok=True)

    def dataset_overview(self) -> dict:
        """Computes basic dataset statistics."""
        stats = {
            "num_samples": len(self.df),
            "num_columns": len(self.df.columns),
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicate_records": int(self.df.duplicated().sum())
        }
        return stats

    def text_analysis(self) -> dict:
        """Analyzes text lengths, vocab, and frequencies."""
        if self.text_col not in self.df.columns:
            return {}
            
        texts = self.df[self.text_col].dropna().astype(str).tolist()
        
        char_counts = [len(t) for t in texts]
        word_counts = [len(t.split()) for t in texts]
        
        all_words = []
        for t in texts:
            all_words.extend(t.lower().split())
            
        vocab = set(all_words)
        word_freq = Counter(all_words)
        
        stats = {
            "avg_char_length": np.mean(char_counts) if char_counts else 0,
            "avg_word_count": np.mean(word_counts) if word_counts else 0,
            "max_word_count": np.max(word_counts) if word_counts else 0,
            "vocab_size": len(vocab),
            "top_10_words": word_freq.most_common(10),
            "bottom_10_words": word_freq.most_common()[-10:] if len(word_freq) >= 10 else []
        }
        return stats

    def quality_analysis(self) -> dict:
        """Identifies noise like URLs, mentions, emojis, and empty strings."""
        if self.text_col not in self.df.columns:
            return {}
            
        texts = self.df[self.text_col].dropna().astype(str)
        
        empty_sentences = (texts.str.strip() == '').sum()
        urls = texts.apply(lambda x: len(re.findall(r'http[s]?://\S+', x))).sum()
        mentions = texts.apply(lambda x: len(re.findall(r'@\w+', x))).sum()
        hashtags = texts.apply(lambda x: len(re.findall(r'#\w+', x))).sum()
        
        # Simple heuristic for Telugu vs Latin characters
        # Telugu unicode range: \u0C00-\u0C7F
        has_telugu = texts.apply(lambda x: bool(re.search(r'[\u0C00-\u0C7F]', x))).sum()
        has_latin = texts.apply(lambda x: bool(re.search(r'[A-Za-z]', x))).sum()
        mixed_scripts = texts.apply(lambda x: bool(re.search(r'[\u0C00-\u0C7F]', x)) and bool(re.search(r'[A-Za-z]', x))).sum()
        
        stats = {
            "empty_sentences": int(empty_sentences),
            "total_urls": int(urls),
            "total_mentions": int(mentions),
            "total_hashtags": int(hashtags),
            "samples_with_telugu_script": int(has_telugu),
            "samples_with_latin_script": int(has_latin),
            "samples_with_mixed_script": int(mixed_scripts)
        }
        return stats

    def generate_visualizations(self):
        """Generates and saves publication-quality plots."""
        if self.text_col not in self.df.columns:
            return
            
        texts = self.df[self.text_col].dropna().astype(str)
        word_counts = texts.apply(lambda x: len(x.split()))
        
        # 1. Sentence Length Histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(word_counts, bins=50, kde=True, color='skyblue')
        plt.title(f'Sentence Length (Word Count) Distribution - {self.dataset_name}')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(os.path.join(self.output_dir, 'sentence_length_dist.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Top Words Bar Chart
        all_words = []
        for t in texts.tolist():
            all_words.extend(t.lower().split())
        word_freq = Counter(all_words)
        
        if word_freq:
            top_words = word_freq.most_common(20)
            words, counts = zip(*top_words)
            plt.figure(figsize=(12, 6))
            sns.barplot(x=list(counts), y=list(words), palette='viridis')
            plt.title(f'Top 20 Most Frequent Words - {self.dataset_name}')
            plt.xlabel('Frequency')
            plt.savefig(os.path.join(self.output_dir, 'top_words.png'), dpi=300, bbox_inches='tight')
            plt.close()
