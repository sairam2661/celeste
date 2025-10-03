import hashlib

def content_hash(*contents: str) -> str:
    """Create a short hash from multiple strings (e.g., grammar + prompt).
    
    Useful for creating unique identifiers for experiments.
    """
    combined = "&?!@&".join(contents)
    return hashlib.md5(combined.encode('utf-8')).hexdigest()[:8]