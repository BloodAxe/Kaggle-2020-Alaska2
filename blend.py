import argparse
import pandas as pd

from alaska2.submissions import blend_predictions_ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submissions", nargs="+", type=str)
    parser.add_argument("-o", "--output", type=str, required=True)
    args = parser.parse_args()

    submissions = [pd.read_csv(x).sort_values(by="Id").reset_index() for x in args.submissions]

    # Force 1.01 value of OOR values in my submission
    # Scripts assumes ABBA's submission goes first
    oor_mask = submissions[0].Label > 1.0
    for s in submissions[1:]:
        s.loc[oor_mask, "Label"] = 1.01

    submissions_blend = blend_predictions_ranked(submissions)

    print(submissions_blend.describe())
    submissions_blend.to_csv(args.output, index=False)
    print("Saved blend to", args.output)


if __name__ == "__main__":
    main()
