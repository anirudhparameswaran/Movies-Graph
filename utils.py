import re

def jaccard_similarity_score(g1: str, g2: str) -> float:
    """Calculates Jaccard similarity for genre strings (e.g., 'Action/Sci-Fi')."""
    
    # Standardize and split genres by '/', ',', and whitespace
    def clean_genres(genre_str):
        if not isinstance(genre_str, str):
            return set()
        genre_str = genre_str.lower()
        # Replace non-word characters (except space) with comma, then split by space or comma
        genres = re.split(r'[/,]', genre_str)
        # Remove empty strings and spaces
        return set(g.strip() for g in genres if g.strip())

    set1 = clean_genres(g1)
    set2 = clean_genres(g2)
    
    if not set1 and not set2:
        return 1.0 # Both empty, perfect match
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union # Jaccard Index: intersection / union