import re
import string

# Indonesian common stopwords
INDONESIAN_STOPWORDS = set([
    "dan", "yang", "di", "ke", "dari", "ini", "itu", "pada", "adalah", "sebagai",
    "dengan", "untuk", "oleh", "atau", "juga", "sudah", "akan", "telah", "bisa",
    "ada", "saja", "saya", "kami", "mereka", "anda", "dia", "kita", "bukan", "hanya",
    "ia", "pun", "agar", "supaya", "hal", "saat", "oleh", "karena", "maka", "jika"
])

def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Clean input review text:
    1. Convert to lowercase
    2. Remove URLs, emails, HTML tags
    3. Remove punctuation and numbers
    4. Normalize whitespace
    5. Optionally filter out Indonesian stopwords
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs, emails, HTML
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    
    # 3. Replace punctuation and numbers with space
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # 4. Split tokens
    tokens = text.split()
    
    # 5. Optional stopword removal
    if remove_stopwords:
        tokens = [t for t in tokens if t not in INDONESIAN_STOPWORDS and len(t) > 1]
    else:
        tokens = [t for t in tokens if len(t) > 1]
        
    cleaned = " ".join(tokens)
    return cleaned
