# Feature Engineering for Sentiment Analysis
# Extract additional features: emoji count, text length, negation patterns
#
# Example usage:
#   extractor = FeatureExtractor()
#   text = "I'm not happy!!! This is terrible 😢"
#   features = extractor.extract_all_features(text)
#   # Output: {
#   #   'emoji_count': 1,
#   #   'exclamation_count': 3,
#   #   'negation_count': 1,  # "not"
#   #   'word_count': 6,
#   #   'char_length': 37,
#   #   ...
#   # }

import re
from typing import Dict, List
import emoji


class FeatureExtractor:
    """
    Extract engineered features from text that may help with sentiment analysis.
    These features complement the word embeddings.
    """
    
    def __init__(self):
        # Common negation words that flip sentiment
        self.negation_words = {
            'not', 'no', 'never', 'neither', 'nobody', 'none', 'nothing',
            'nowhere', 'hardly', 'barely', 'scarcely', "n't", 'cannot', "can't"
        }
        
    def count_emojis(self, text: str) -> int:
        """
        Count the number of emoji characters in text.
        Emojis are strong sentiment indicators in social media.
        
        Args:
            text: Input text string
            
        Returns:
            Number of emojis found
        """
        # Simple emoji detection using common emoji ranges
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        
        emojis = emoji_pattern.findall(text)
        return len(emojis)
    
    def count_exclamations(self, text: str) -> int:
        """
        Count exclamation marks - often indicate strong sentiment.
        
        Args:
            text: Input text string
            
        Returns:
            Number of exclamation marks
        """
        return text.count('!')
    
    def count_questions(self, text: str) -> int:
        """
        Count question marks.
        
        Args:
            text: Input text string
            
        Returns:
            Number of question marks
        """
        return text.count('?')
    
    def count_capitals(self, text: str) -> int:
        """
        Count uppercase words - may indicate shouting or emphasis.
        
        Args:
            text: Input text string
            
        Returns:
            Number of fully capitalized words
        """
        words = text.split()
        # Count words that are all uppercase and longer than 1 character
        caps = [w for w in words if w.isupper() and len(w) > 1]
        return len(caps)
    
    def count_negations(self, text: str) -> int:
        """
        Count negation words that may flip sentiment.
        Example: "not good" has negative sentiment despite "good"
        
        Args:
            text: Input text string
            
        Returns:
            Number of negation words found
        """
        words = text.lower().split()
        negation_count = sum(1 for word in words if word in self.negation_words)
        return negation_count
    
    def get_text_length(self, text: str) -> int:
        """
        Get character length of text.
        Longer reviews may be more detailed/thoughtful.
        
        Args:
            text: Input text string
            
        Returns:
            Number of characters
        """
        return len(text)
    
    def get_word_count(self, text: str) -> int:
        """
        Get word count of text.
        
        Args:
            text: Input text string
            
        Returns:
            Number of words
        """
        return len(text.split())
    
    def extract_all_features(self, text: str) -> Dict[str, int]:
        """
        Extract all engineered features from text.
        Returns a dictionary of feature name -> value.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary of feature names and values
        """
        features = {
            'emoji_count': self.count_emojis(text),
            'exclamation_count': self.count_exclamations(text),
            'question_count': self.count_questions(text),
            'capital_word_count': self.count_capitals(text),
            'negation_count': self.count_negations(text),
            'char_length': self.get_text_length(text),
            'word_count': self.get_word_count(text),
        }
        return features
    
    def extract_features_batch(self, texts: List[str]) -> List[Dict[str, int]]:
        """
        Extract features for multiple texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of feature dictionaries
        """
        return [self.extract_all_features(text) for text in texts]


def get_avg_word_length(text: str) -> float:
    """
    Calculate average word length in text.
    Longer words may correlate with more formal/detailed reviews.
    
    Args:
        text: Input text string
        
    Returns:
        Average word length
    """
    words = text.split()
    if not words:
        return 0.0
    
    total_length = sum(len(word) for word in words)
    return total_length / len(words)
