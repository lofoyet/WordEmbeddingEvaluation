# -*- coding: utf-8 -*-
"""Text utilities."""
import re

from nltk.corpus import stopwords

# recompiled into set for faster performance
STOPS = set(stopwords.words("english"))


def clean_text(text):
    """Clean."""
    text = text.lower()

    # Format words and remove unwanted characters
    # remove URl
    text = re.sub(r"https?:\/\/.*[\r\n]*", "", text, flags=re.MULTILINE)
    # fix ampersand
    text.replace("&amp;", "")
    # number should be removed
    text = re.sub(r"\d+", "number", text)
    # punctuations
    text = re.sub(r"""[_"\-;%()|+&=*%,.!?:#$@â£\[\]/]""", " ", text)

    # remove stop words
    # punctuaions already removed, safe to split on space
    text = text.split()
    text = [w for w in text if w not in STOPS]
    text = " ".join(text)

    return text
