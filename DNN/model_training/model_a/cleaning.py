# Text cleaning utilities
# Remove noise, normalize text, handle special characters
#
# Example usage:
#   cleaner = TextCleaner()
#   text = "Check this out!!! https://example.com <br> AMAZING 😊"
#   cleaned = cleaner.clean(text)
#   # Output: "check this out!!! amazing 😊"
#   # (URLs and HTML removed, lowercased, emoji preserved)

import re
import string
from typing import List


class TextCleaner:
    """
    Cleans and normalizes text data for sentiment analysis.
    Handles common noise in social media and review text.
    """
    
    def __init__(self, lowercase: bool = True, remove_urls: bool = True, 
                 remove_html: bool = True, remove_special_chars: bool = False):
        """
        Initialize the text cleaner with configuration options.
        
        Args:
            lowercase: Convert all text to lowercase
            remove_urls: Remove HTTP/HTTPS URLs from text
            remove_html: Remove HTML tags
            remove_special_chars: Remove special characters (keep only alphanumeric)
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.remove_special_chars = remove_special_chars
        
    def clean(self, text: str) -> str:
        """
        Apply all cleaning steps to input text.
        
        Args:
            text: Raw input text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs (e.g., https://example.com)
        if self.remove_urls:
            text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove HTML tags (e.g., <br>, <div>)
        if self.remove_html:
            text = re.sub(r'<.*?>', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove extra whitespace and trim
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Optionally remove special characters (keeps emoji if False)
        if self.remove_special_chars:
            text = re.sub(f'[^a-zA-Z0-9\s]', '', text)
        
        return text
    
    def remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation marks from text.
        Useful for bag-of-words models, but may hurt sentiment (e.g., "!!!")
        
        Args:
            text: Input text
            
        Returns:
            Text without punctuation
        """
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def normalize_repeated_chars(self, text: str, max_repeats: int = 2) -> str:
        """
        Reduce repeated characters (e.g., "loooove" -> "loove").
        Common in social media text for emphasis.
        
        Args:
            text: Input text
            max_repeats: Maximum allowed character repetitions
            
        Returns:
            Text with normalized repetitions
        """
        pattern = re.compile(r'(.)\1{' + str(max_repeats) + r',}')
        return pattern.sub(r'\1' * max_repeats, text)
    
    def batch_clean(self, texts: List[str]) -> List[str]:
        """
        Clean multiple texts efficiently.
        
        Args:
            texts: List of raw text strings
            
        Returns:
            List of cleaned text strings
        """
        return [self.clean(text) for text in texts]


def preprocess_for_sentiment(text: str) -> str:
    """
    Quick preprocessing function for sentiment analysis.
    Keeps emoji and punctuation since they're important for sentiment.
    
    Args:
        text: Raw input text
        
    Returns:
        Preprocessed text ready for tokenization
    """
    cleaner = TextCleaner(lowercase=True, remove_urls=True, 
                         remove_html=True, remove_special_chars=False)
    
    # Clean the text
    text = cleaner.clean(text)
    
    # Normalize repeated characters (e.g., "soooo good" -> "sooo good")
    text = cleaner.normalize_repeated_chars(text, max_repeats=3)
    
    return text

