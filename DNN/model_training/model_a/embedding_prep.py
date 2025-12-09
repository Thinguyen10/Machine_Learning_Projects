# Embedding preparation utilities
# Load pre-trained embeddings (GloVe, FastText) and prepare embedding matrices
#
# Example usage:
#   loader = EmbeddingLoader(embedding_dim=100)
#   loader.load_glove('glove.6B.100d.txt')
#   # Loaded 400000 word embeddings
#   
#   word_to_idx = {'<PAD>': 0, '<UNK>': 1, 'love': 2, 'hate': 3}
#   embedding_matrix = loader.create_embedding_matrix(word_to_idx, vocab_size=4)
#   # Creates matrix of shape (4, 100) with pre-trained vectors for 'love' and 'hate'
#   
#   similar = loader.most_similar('good', top_k=3)
#   # Output: [('great', 0.82), ('excellent', 0.79), ('nice', 0.75)]

import numpy as np
from typing import Dict, List, Tuple


class EmbeddingLoader:
    """
    Loads pre-trained word embeddings and creates embedding matrices
    for our vocabulary.
    """
    
    def __init__(self, embedding_dim: int = 100):
        """
        Initialize embedding loader.
        
        Args:
            embedding_dim: Dimension of word vectors (e.g., 100, 200, 300)
        """
        self.embedding_dim = embedding_dim
        self.embeddings: Dict[str, np.ndarray] = {}
        
    def load_glove(self, filepath: str) -> int:
        """
        Load GloVe embeddings from file.
        GloVe format: word vec1 vec2 vec3 ... vecN
        
        Args:
            filepath: Path to GloVe file (e.g., glove.6B.100d.txt)
            
        Returns:
            Number of embeddings loaded
        """
        print(f"Loading GloVe embeddings from {filepath}...")
        count = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                # Convert string numbers to float array
                vector = np.array(values[1:], dtype='float32')
                
                # Verify dimension matches
                if len(vector) == self.embedding_dim:
                    self.embeddings[word] = vector
                    count += 1
        
        print(f"Loaded {count} word embeddings")
        return count
    
    def load_fasttext(self, filepath: str) -> int:
        """
        Load FastText embeddings from file.
        Similar format to GloVe but may include subword information.
        
        Args:
            filepath: Path to FastText file
            
        Returns:
            Number of embeddings loaded
        """
        print(f"Loading FastText embeddings from {filepath}...")
        count = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            # Skip first line if it contains metadata (vocab_size, dimension)
            first_line = f.readline().split()
            if len(first_line) == 2:
                # First line is metadata, continue to actual embeddings
                pass
            else:
                # First line is an embedding, process it
                word = first_line[0]
                vector = np.array(first_line[1:], dtype='float32')
                if len(vector) == self.embedding_dim:
                    self.embeddings[word] = vector
                    count += 1
            
            # Process remaining lines
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                
                if len(vector) == self.embedding_dim:
                    self.embeddings[word] = vector
                    count += 1
        
        print(f"Loaded {count} word embeddings")
        return count
    
    def create_embedding_matrix(self, word_to_idx: Dict[str, int], 
                               vocab_size: int) -> np.ndarray:
        """
        Create embedding matrix for our vocabulary.
        Maps each word index to its pre-trained embedding vector.
        
        Args:
            word_to_idx: Dictionary mapping words to indices
            vocab_size: Size of vocabulary
            
        Returns:
            Numpy array of shape (vocab_size, embedding_dim)
        """
        print(f"Creating embedding matrix for vocab size {vocab_size}...")
        
        # Initialize with random small values
        embedding_matrix = np.random.randn(vocab_size, self.embedding_dim).astype('float32') * 0.01
        
        # Set padding vector to zeros
        embedding_matrix[0] = np.zeros(self.embedding_dim)
        
        found_count = 0
        # Fill in pre-trained embeddings for words we have
        for word, idx in word_to_idx.items():
            if word in self.embeddings:
                embedding_matrix[idx] = self.embeddings[word]
                found_count += 1
        
        coverage = (found_count / vocab_size) * 100
        print(f"Found pre-trained embeddings for {found_count}/{vocab_size} words ({coverage:.2f}%)")
        
        return embedding_matrix
    
    def get_embedding(self, word: str) -> np.ndarray:
        """
        Get embedding vector for a single word.
        Returns random vector if word not found.
        
        Args:
            word: Word to get embedding for
            
        Returns:
            Embedding vector
        """
        if word in self.embeddings:
            return self.embeddings[word]
        else:
            # Return random vector for unknown words
            return np.random.randn(self.embedding_dim).astype('float32') * 0.01
    
    def most_similar(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar words based on cosine similarity.
        Useful for exploring embeddings and understanding word relationships.
        
        Args:
            word: Query word
            top_k: Number of similar words to return
            
        Returns:
            List of (word, similarity_score) tuples
        """
        if word not in self.embeddings:
            print(f"Word '{word}' not found in embeddings")
            return []
        
        query_vec = self.embeddings[word]
        # Normalize query vector
        query_vec_norm = query_vec / np.linalg.norm(query_vec)
        
        similarities = []
        for w, vec in self.embeddings.items():
            if w != word:
                # Calculate cosine similarity
                vec_norm = vec / np.linalg.norm(vec)
                similarity = np.dot(query_vec_norm, vec_norm)
                similarities.append((w, float(similarity)))
        
        # Sort by similarity (highest first) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


def initialize_random_embeddings(vocab_size: int, embedding_dim: int) -> np.ndarray:
    """
    Create random embedding matrix when pre-trained embeddings aren't available.
    Uses Xavier initialization for better training.
    
    Args:
        vocab_size: Number of words in vocabulary
        embedding_dim: Dimension of embedding vectors
        
    Returns:
        Random embedding matrix
    """
    # Xavier initialization: scale by sqrt(1/embedding_dim)
    scale = np.sqrt(1.0 / embedding_dim)
    embeddings = np.random.randn(vocab_size, embedding_dim).astype('float32') * scale
    
    # Set padding (index 0) to zeros
    embeddings[0] = np.zeros(embedding_dim)
    
    return embeddings
