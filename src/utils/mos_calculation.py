import argparse

from wvmos import get_wvmos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predicts-dir', type=str, help='directory with predictions')
    args = parser.parse_args()

    model = get_wvmos(cuda=True)

    mos = model.calculate_dir(args.predicts_dir, mean=True)

    print(mos)
