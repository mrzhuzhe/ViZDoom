import sys
from torch import multiprocessing as mp

def main():
    """Left here for legacy reasons. Use the sample_factory.run_algorithm script from the root folder instead."""
    from sample_factory.run_algorithm import main as run_algorithm_main

    return run_algorithm_main()


if __name__ == '__main__':
    #mp.set_start_method("spawn")
    sys.exit(main())
