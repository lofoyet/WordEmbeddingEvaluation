"""Text utilities."""
import re

from nltk.corpus import stopwords

STOPS = set(stopwords.words("english"))


def clean_text(text):
    """Clean."""
    text = text.lower()

    # Format words and remove unwanted characters
    text = re.sub(r"https?:\/\/.*[\r\n]*", "", text, flags=re.MULTILINE)
    text.replace("&amp;", "")
    text = re.sub(r"\d+", "number", text)
    text = re.sub(r"""[_"\-;%()|+&=*%,.!?:#$@â£\[\]/]""", " ", text)

    # remove stop words
    text = text.split()
    text = [w for w in text if w not in STOPS]
    text = " ".join(text)

    return text
