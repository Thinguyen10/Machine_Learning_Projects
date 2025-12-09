# Model A: Tokenizer and text cleaning
# Preprocessing utilities for sentiment analysis
#
# Example usage:
#   tokenizer = SimpleTokenizer(max_vocab_size=10000)
#   tokenizer.build_vocab(["I love this", "I hate that", "This is great"])
#   encoded = tokenizer.encode("I love this")
#   # Output: [2, 3, 4] (word indices)
#   padded = tokenizer.encode_batch(["I love this"], max_length=10)
#   # Output: [[2, 3, 4, 0, 0, 0, 0, 0, 0, 0]] (padded with zeros)

from typing import List, Dict
import re


class SimpleTokenizer:
    """
    Simple word-level tokenizer for text processing.
    Converts text into sequences of tokens (words).
    """
    
    def __init__(self, max_vocab_size: int = 10000, min_freq: int = 2):
        """
        Initialize tokenizer with vocabulary constraints.
        
        Args:
            max_vocab_size: Maximum number of words to keep in vocabulary
            min_freq: Minimum frequency for a word to be included
        """
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.word_freq: Dict[str, int] = {}
        
        # Special tokens for padding and unknown words
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        
    def tokenize(self, text: str) -> List[str]:
        """
        Split text into individual tokens (words).
        Handles basic punctuation and whitespace.
        
        Args:
            text: Input text string
            
        Returns:
            List of token strings
        """
        # Split on whitespace and basic punctuation
        # Keep common punctuation that affects sentiment (!?.)
        tokens = re.findall(r'\b\w+\b|[!?.]', text.lower())
        return tokens
    
    def build_vocab(self, texts: List[str]):
        """
        Build vocabulary from a corpus of texts.
        Counts word frequencies and creates word-to-index mappings.
        
        Args:
            texts: List of text documents
        """
        # Count word frequencies across all texts
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                self.word_freq[token] = self.word_freq.get(token, 0) + 1
        
        # Filter words by minimum frequency
        filtered_words = [(word, freq) for word, freq in self.word_freq.items() 
                         if freq >= self.min_freq]
        
        # Sort by frequency (most common first)
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        
        # Take top max_vocab_size words
        filtered_words = filtered_words[:self.max_vocab_size - 2]  # -2 for PAD and UNK
        
        # Create word-to-index mapping
        # Reserve index 0 for padding, index 1 for unknown
        self.word_to_idx = {self.PAD_TOKEN: self.PAD_IDX, 
                           self.UNK_TOKEN: self.UNK_IDX}
        
        for idx, (word, _) in enumerate(filtered_words, start=2):
            self.word_to_idx[word] = idx
        
        # Create reverse mapping (index-to-word)
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        print(f"Vocabulary built: {len(self.word_to_idx)} words")
        print(f"Most common words: {filtered_words[:10]}")
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to sequence of integer indices.
        
        Args:
            text: Input text string
            
        Returns:
            List of integer token indices
        """
        tokens = self.tokenize(text)
        # Map each token to its index, use UNK_IDX for unknown words
        indices = [self.word_to_idx.get(token, self.UNK_IDX) for token in tokens]
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """
        Convert sequence of indices back to text.
        
        Args:
            indices: List of token indices
            
        Returns:
            Reconstructed text string
        """
        words = [self.idx_to_word.get(idx, self.UNK_TOKEN) for idx in indices]
        # Filter out padding tokens
        words = [w for w in words if w != self.PAD_TOKEN]
        return ' '.join(words)
    
    def encode_batch(self, texts: List[str], max_length: int = None) -> List[List[int]]:
        """
        Encode multiple texts and optionally pad to same length.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length (None = no padding)
            
        Returns:
            List of encoded sequences
        """
        encoded = [self.encode(text) for text in texts]
        
        # Pad sequences to max_length if specified
        if max_length is not None:
            padded = []
            for seq in encoded:
                if len(seq) < max_length:
                    # Pad with PAD_IDX to reach max_length
                    seq = seq + [self.PAD_IDX] * (max_length - len(seq))
                else:
                    # Truncate if longer than max_length
                    seq = seq[:max_length]
                padded.append(seq)
            return padded
        
        return encoded
    
    def get_vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.word_to_idx)

