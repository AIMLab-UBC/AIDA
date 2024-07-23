from utils.parser import parse_arguments
from Train import Train

def main():
    cfg = parse_arguments()
    t = Train(cfg)
    t.run()

if __name__ == "__main__":
    main()
