r"""Evaluate word embedding scripts.

python evaluate_word_embedding.py \
    --out_dir=output/eval\
    --emb_path=/tmp/glove.twitter.27B.100d.txt\
    --csv_separator=" "\
    --quoting=3
"""
import os
import sys

import click

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from lib.word_embedding_evaluator import Scheduler  # noqa


@click.command()
@click.option(
    "--out_dir",
    type=str,
    help="Dir to output evaluation",
    required=True)
@click.option(
    "--emb_path",
    type=str,
    help="The word embedding data path",
    required=True)
@click.option(
    "--csv_separator",
    type=str,
    help="Separator for embedding file",
    required=True)
@click.option(
    "--quoting",
    type=int,
    help="Quoting schema when reading embedding in pandas",
    required=True)
@click.option(
    "--evaluators",
    type=str,
    help="Tasks to run, comma-separate each task",
    default="word_similarity_353",
    required=True)
def word_embedding_evaluate(
        out_dir, emb_path, csv_separator, quoting, evaluators):
    """Main body of function."""
    evaluators = evaluators.split(",")
    scheduler = Scheduler(
        out_dir, emb_path, csv_separator, quoting)
    scheduler.add_tasks(*evaluators)
    scheduler.run()


if __name__ == "__main__":
    word_embedding_evaluate()
