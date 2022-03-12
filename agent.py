import gym
from wordle_rl import WordleEnv


class Wordler(object):
    def __init__(self):
        self.env = gym.make(WordleEnv).env
        self.env.seed(19)


def main():
    agent = Wordler()
    return None


if __name__ == "__main__":
    main()
