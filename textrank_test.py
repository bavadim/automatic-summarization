#!/usr/bin/env python3

import unittest
import tensorflow as tf
from textrank import USETextRank

class TestUSETextRank(unittest.TestCase):

    def setUp(self):
        self.summarizer = USETextRank()

    def test_sim_mat(self):
        g = [
            [1, 0.],
            [1, 0.],
            [1., 2.],
            [1., 2.],
            [2., 2.],
            ]

        res = tf.constant([
            [1.,      1.,      0.44721, 0.44721, 0.70711],
            [1.,     1.,      0.44721, 0.44721, 0.70711],
            [0.44721, 0.44721, 1.,      1.,      0.94868],
            [0.44721, 0.44721, 1.,      1.,      0.94868],
            [0.70711, 0.70711, 0.94868, 0.94868, 1.]
        ])

        self.assertTrue(tf.reduce_all(tf.math.equal(self.summarizer.__sim_mat__(g), res)).numpy())

    def test_ranks(self):
        g = [
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0, 0, 1],
        ]

        res = tf.constant([0.,      0.,      1.])

        self.assertTrue(tf.reduce_all(tf.math.equal(self.summarizer.__ranks__(g), res)).numpy())

    def test_the_most_important(self):
        text = 'Here is an example to better understand the notation above. We have a graph to represent how web pages link to each other. Each node represents a webpage, and the arrows represent edges. We want to get the weight of webpage e.'
        sents = self.summarizer.the_most_important(text, 20)
        res = [
            'We have a graph to represent how web pages link to each other.',
            'We want to get the weight of webpage e.',
            'Here is an example to better understand the notation above.',
            'Each node represents a webpage, and the arrows represent edges.'
        ]
        self.assertEqual(sents, res)


if __name__ == '__main__':
    unittest.main()
