"""Install and setup."""
from setuptools import setup

setup(
    name="WordEmbeddingEvaluation",
    version="0.2.0",
    packages=["word_embedding_evaluation"],
    install_requires=[
        "Click>=6.7",
        "nltk>=3.2.5",
        "scikit-learn>=0.19.0",
        "pandas>=0.23.0"
    ],
    entry_points={
        "console_scripts": [
            "WordEmbEval=word_embedding_evaluation.__main__:main"
        ]
    },
    description="Evaluation tasks on word embedding",
    author="Liu,Tianpei",
    author_email="tianpei.liu0@gmail.com",
)
