"""This is an evaluator qwen v1 here."""
import numpy as np
import os
import random
import json

from typing import Any, Dict, List

def read_json(filename):
    """Reads json file of a network."""
    with open(filename, 'r') as fr:
        data = json.loads(fr.read())
    return data


def shuffle_jointly(
        data: list[tuple[int, int]],
        labels:list[int]
) -> tuple[list[tuple[int, int]], list[int]]:
    """Jointly shuffles train/test data."""
    combined = list(zip(data, labels))
    random.shuffle(combined)
    new_data, new_labels = map(list, zip(*combined))
    return new_data, new_labels


def prep_data(data: dict) -> tuple:
    '''Unwraps the data.'''
    train, tr_labels = shuffle_jointly(data['train_edges'], data['train_labels'])
    test, test_labels = shuffle_jointly(data['test_edges'], data['test_labels'])
    return data['nodes'], train, tr_labels, test, test_labels


def read_data(filename: str, folder: str) -> tuple:
    """Main evaluation function."""
    data = read_json(os.path.join(folder, filename))
    nodes, train_edge, train_labels, test_edge, test_labels = prep_data(data)
    print('Nodes ', len(nodes), 'Train ', len(train_labels), ' Classes ', sum(train_labels), len(train_labels), ' Test ', sum(test_labels), len(test_labels))
    return nodes, train_edge, train_labels, test_edge, test_labels


if __name__ == "__main__":
    """Demonstrates reading of the network data."""

    data_path = '../550nets'
    filename = 'bleather_plant_herbivore_webs_britain_00.json'
    nodes, train_edge, train_labels, test_edge, test_labels = read_data(filename, data_path)

