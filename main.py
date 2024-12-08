import os
import Model.GenerativeTraining as GenerativeTraining
import Model.ABGTraining as ABGTraining

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    GenerativeTraining.main(ROOT_DIR)
    ABGTraining.main(ROOT_DIR)

if __name__ == "__main__":
    main()