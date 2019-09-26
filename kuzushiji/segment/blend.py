import argparse


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('models', nargs='+')
    arg('output')
    args = parser.parse_args()


if __name__ == '__main__':
    main()
