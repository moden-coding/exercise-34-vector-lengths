#!/usr/bin/env python3

import numpy as np

def vector_lengths(a):
    return np.array([])

def main():
    test_array = np.array([[1, 2, 3], [4, 5, 6]])
    lengths = vector_lengths(test_array)
    print(f"Row vector lengths of\n{test_array}:\n{lengths}")

if __name__ == "__main__":
    main()
