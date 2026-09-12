#!/usr/bin/env python3

import timeit
import unittest
from unittest.mock import patch

import numpy as np
import scipy.linalg

from src.vector_lengths import main, vector_lengths


class TestVectorLengths(unittest.TestCase):

    def test_content(self):
        n = 10
        for m in range(1, 5):
            a = np.random.randn(n, m)
            v = vector_lengths(a)
            self.assertEqual(v.shape, (n,), msg="Incorrect shape!")
            correct = scipy.linalg.norm(a, 2, axis=1)
            np.testing.assert_allclose(
                v, correct, err_msg="Incorrect result for matrix:\n%s" % a)

    def test_performance(self):
        a = np.random.randint(0, 10000, (300, 30))
        t = timeit.timeit(
            stmt="vector_lengths(a)",
            globals={"a": a, "vector_lengths": vector_lengths},
            number=1000)
        self.assertLess(
            t, 0.1,
            msg="Your code runs slow. Are you sure you use vectorized "
                "operations?")

    def test_main_calls_vector_lengths(self):
        with patch("src.vector_lengths.vector_lengths", wraps=vector_lengths) as vl:
            main()
        self.assertGreaterEqual(
            vl.call_count, 1,
            msg="You should call the vector_lengths function from the main "
                "function!")

    def test_no_scipy(self):
        a = np.random.randn(10, 3)
        try:
            with patch("src.vector_lengths.scipy.linalg.norm") as pnorm:
                vector_lengths(a)
                pnorm.assert_not_called()
        except (AttributeError, ModuleNotFoundError):
            # src.vector_lengths only has a "scipy" attribute to patch if the
            # student's solution imports scipy.linalg itself; if it doesn't,
            # there is nothing to check here.
            pass

    def test_single_component_vectors(self):
        a = np.array([[3.0], [-4.0], [0.0]])
        v = vector_lengths(a)
        np.testing.assert_allclose(
            v, [3.0, 4.0, 0.0],
            err_msg="For single-column input, the length of each row is "
                    "just its absolute value. vector_lengths(%r) should be "
                    "[3.0, 4.0, 0.0]." % a)


if __name__ == '__main__':
    unittest.main()
