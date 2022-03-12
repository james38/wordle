import json
import math
import random
import string

import numpy as np
import pandas as pd

import gym
from gym import spaces, utils, wrappers

import torch
import torch.nn as nn


class WordleEnv(gym.Env):

    def __init__(
        self,
        secret_word=None,
        valid_words_loc='./valids.json',
        solution_words_loc='./words.json',
        n_letters=5,
        max_attempts=6,
        L=3.4,
    ):
        self.seed()

        self.load_words(valid_words_loc)
        # construct the possible actions as integer-indexed valid words
        self.action_words = [(i, w) for i, w in enumerate(self.words)]

        assert type(n_letters) is int, "number of letters must be int"
        assert type(max_attempts) is int, "number of max attempts must be int"
        self.n_letters = n_letters
        self.max_attempts = max_attempts
        self.L = L
        self.gen_reward()
        self.reward_range = (0,1)

        alphabet = list(string.ascii_lowercase)
        self.len_alphabet = len(alphabet)
        self.alphabet_order = {letter: i for i, letter in enumerate(alphabet)}

        # first there is a discrete input for how many turns have elapsed,
        #  then each letter is represented by 7 inputs
        # of the 7, there are 5 positional inputs per letter, where the
        #  value 2 represents green, 1 as possible yellow, and 0 otherwise
        # the last 2 inputs per letter are metadata, one bit and one discrete
        #  if the first bit is 1, then it has been guessed a number of times
        #   that exceed the number of times it is present in the solution,
        #   (eg. this can include letters that are yellow or green)
        #  the second discrete input is the maximum number of yellows seen
        #   for any guess for that letter (up to 2 for a 5 letter guess)
        # this is a required variable for the environment
        self.observation_space = spaces.Tuple(
            (
                spaces.MultiDiscrete(
                    [self.max_attempts]
                    + self.len_alphabet * [3] * self.n_letters
                ),
                spaces.MultiBinary(self.len_alphabet),
                spaces.MultiDiscrete(
                    self.len_alphabet * [1 + int(self.n_letters / 2)]
                ),
            )
        )
        self.base_obs = np.zeros(
            1
            + self.len_alphabet * self.n_letters
            + 2 * self.len_alphabet
        )

        # required variable for environment
        self.action_space = spaces.Discrete(len(self.words))

        self.load_solutions(solution_words_loc)
        if secret_word is None:
            self.reset()
        else:
            assert type(secret_word) is str, "secret word must be str"
            assert len(secret_word) == n_letters, "secret word wrong length"
            self.reset(secret_word)


    def load_words(self, valid_words_loc):
        with open(valid_words_loc) as f:
            data = f.read()
        self.words = json.loads(data)
        return None


    def load_solutions(self, solution_words_loc):
        with open(solution_words_loc) as f:
            words = json.load(f)
        random.shuffle(words['solutions'])
        self.poss_solutions = words['solutions']
        return None


    def get_random_secret_word(self):
        return random.choice(self.poss_solutions)


    def reset(self, secret_word=None):
        """
        restarts the environment for a new episode
        returns the starting state as a value of self.observation_space
        """
        if secret_word is None:
            self.secret_word = self.get_random_secret_word()
        else:
            self.secret_word = secret_word
        self.n_guesses = 0
        self.incorrect_letters = list()

        self.state = self.base_obs.copy()
        self.reward = 0
        self.done = False
        self.info = dict()

        return self.state


    def step(self, action):
        """
        takes as input action within self.action_space
        the environment responds with the resultant state
        a 4 tuple of:
          state - same type as reset method return
          reward - a return value received from transitioning to the new state
          done - a boolean for whether the episode is over
          info - an optional dictionary that may be useful to populate
        """
        assert self.action_space.contains(action)
        guess = self.words[action]

        # hard mode constraint
        for j in guess:
            if j in self.incorrect_letters:
                return [self.state, self.reward, self.done, self.info]

            breakpoint()
            j_is_exceeded = self.state[
                2*(self.alphabet_order[j] - self.len_alphabet)
            ]
            j_max_yellows = self.state[
                (self.alphabet_order[j] - self.len_alphabet)
            ]

            # if ():
            #     return [self.state, self.reward, self.done, self.info]

        self.n_guesses += 1
        self.state[0] = self.n_guesses

        if guess == self.secret_word:
            self.reward = self.rewards[self.n_guesses - 1]
            self.done = True

        if self.n_guesses == self.max_attempts:
            self.done = True

        for i,j in enumerate(guess):
            if j == self.secret_word[i]:
                self.state[
                    1 + self.alphabet_order[j] * self.n_letters + i
                ] = 2
            elif j in self.secret_word:
                letter_state = self.state[
                    (
                        1 + self.alphabet_order[j] * self.n_letters
                    ):(
                        1 + self.alphabet_order[j] * self.n_letters
                        + self.n_letters
                    )
                ]
            else:
                pass

        # for j in guess:
        #     self.incorrect_letters.append(j)

        return [self.state, self.reward, self.done, self.info]


    def gt_n_poisson(self, L, n):
        """
        returns the inverse of the Poisson CDF with mean, L, at x=n integer
        """
        assert type(n) is int, "sample value must be integer"
        sum_prob = 0
        for i in range(n):
            sum_prob += (L**i) / math.factorial(i)
        sum_prob *= np.exp(-L)
        return 1 - sum_prob


    def gen_reward(self):
        """
        returns the reward structure that provides return in proportion to
        the number of guesses for success
        """
        self.rewards = [
            self.gt_n_poisson(self.L, i)
            for i in range(1, 1 + self.max_attempts)
        ]
        return None


    def gen_possible_curr_board(self):
        """
        *unused idea*
        calculate total possible board states
        [65, 80, 50, 20, 5]
          for (0, 1, 2, 3, 4) correct placings
          and all possible numbers of correct letters incorrectly placed
           including no correct letters incorrectly placed
        """
        self.board = [
            (
                math.comb(self.n_letters, n)
                * sum(
                    [
                        math.perm(self.n_letters - n - 1, j)
                        for j in range(self.n_letters - n)
                    ]
                )
            ) for n in range(self.n_letters)
        ]
        return None


class Wordler(object):
    def __init__(self):
        self.env = gym.make(WordleEnv).env
        self.env.seed(19)


def main():
    wordle = WordleEnv()
    return None


if __name__ == "__main__":
    main()
