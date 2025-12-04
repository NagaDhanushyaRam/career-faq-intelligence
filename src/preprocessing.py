"""
Text Preprocessing Module
Implements text normalization, cleaning, tokenization, and stopword filtering.
Based on Listing 2 from the Midterm Report.
"""
import re
import string
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK resources."""
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {resource}: {e}")

# Initialize NLTK data
download_nltk_data()

# Initialize tools
try:
    STOP_WORDS = set(stopwords.words('english'))
except:
    STOP_WORDS = set()
    
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()


class TextPreprocessor:
    """
    Text preprocessing class for FAQ recommendation system.
    
    Features:
    - Text normalization (lowercase, unicode handling)
    - Cleaning (remove special characters, URLs, emails)
    - Tokenization
    - Stopword removal
    - Lemmatization/Stemming
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        stem: bool = False,
        min_token_length: int = 2,
        custom_stopwords: Optional[List[str]] = None
    ):
        """
        Initialize the preprocessor with configuration options.
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation characters
            remove_numbers: Remove numeric characters
            remove_stopwords: Remove common English stopwords
            lemmatize: Apply lemmatization (recommended)
            stem: Apply stemming (alternative to lemmatization)
            min_token_length: Minimum length of tokens to keep
            custom_stopwords: Additional stopwords to remove
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stem = stem
        self.min_token_length = min_token_length
        
        # Build stopwords set
        self.stopwords = STOP_WORDS.copy()
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
    
    def normalize(self, text: str) -> str:
        """
        Normalize text: lowercase and handle unicode.
        """
        if not isinstance(text, str):
            text = str(text)
        
        if self.lowercase:
            text = text.lower()
        
        # Normalize unicode characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        return text
    
    def clean(self, text: str) -> str:
        """
        Clean text by removing unwanted patterns.
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation (optional)
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers (optional)
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        """
        try:
            tokens = word_tokenize(text)
        except Exception:
            # Fallback to simple split
            tokens = text.split()
        
        return tokens
    
    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens: remove stopwords and short tokens.
        """
        filtered = []
        
        for token in tokens:
            # Skip short tokens
            if len(token) < self.min_token_length:
                continue
            
            # Skip stopwords
            if self.remove_stopwords and token.lower() in self.stopwords:
                continue
            
            filtered.append(token)
        
        return filtered
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply lemmatization to tokens.
        """
        if self.lemmatize:
            return [LEMMATIZER.lemmatize(token) for token in tokens]
        return tokens
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming to tokens.
        """
        if self.stem:
            return [STEMMER.stem(token) for token in tokens]
        return tokens
    
    def preprocess(self, text: str, return_tokens: bool = False) -> str | List[str]:
        """
        Apply full preprocessing pipeline.
        
        Args:
            text: Input text to preprocess
            return_tokens: If True, return list of tokens; else return joined string
            
        Returns:
            Preprocessed text as string or list of tokens
        """
        # Step 1: Normalize
        text = self.normalize(text)
        
        # Step 2: Clean
        text = self.clean(text)
        
        # Step 3: Tokenize
        tokens = self.tokenize(text)
        
        # Step 4: Filter
        tokens = self.filter_tokens(tokens)
        
        # Step 5: Lemmatize or Stem
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        elif self.stem:
            tokens = self.stem_tokens(tokens)
        
        if return_tokens:
            return tokens
        
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str], return_tokens: bool = False) -> List:
        """
        Preprocess a batch of texts.
        """
        return [self.preprocess(text, return_tokens) for text in texts]


# Create default preprocessor instance
default_preprocessor = TextPreprocessor()


def preprocess_text(text: str) -> str:
    """Convenience function for quick preprocessing."""
    return default_preprocessor.preprocess(text)


def preprocess_for_tfidf(text: str) -> str:
    """Preprocess text optimized for TF-IDF (no lemmatization for exact matching)."""
    preprocessor = TextPreprocessor(lemmatize=False, stem=False)
    return preprocessor.preprocess(text)


def preprocess_for_sbert(text: str) -> str:
    """Preprocess text for Sentence-BERT (minimal cleaning to preserve semantics)."""
    preprocessor = TextPreprocessor(
        remove_stopwords=False,  # Keep stopwords for semantic context
        lemmatize=False,
        remove_punctuation=False
    )
    return preprocessor.preprocess(text)


if __name__ == "__main__":
    # Test the preprocessor
    test_texts = [
        "How do I write a good resume for my first job?",
        "What are the BEST interview tips for freshers???",
        "Tell me about salary negotiation strategies!!! Check out https://example.com",
        "What's the difference between CV and Resume?"
    ]
    
    print("="*60)
    print("TEXT PREPROCESSING DEMO")
    print("="*60)
    
    preprocessor = TextPreprocessor()
    
    for text in test_texts:
        processed = preprocessor.preprocess(text)
        tokens = preprocessor.preprocess(text, return_tokens=True)
        
        print(f"\nOriginal: {text}")
        print(f"Processed: {processed}")
        print(f"Tokens: {tokens}")

