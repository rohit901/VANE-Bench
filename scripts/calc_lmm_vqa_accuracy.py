import glob
import argparse

def main(path):
    pred_files = glob.glob(path + "/*.txt")
    pred_files = sorted(pred_files, key = lambda x: (int(x.split("/")[-1].split('_')[1]), int(x.split("/")[-1].split('_')[3]))) # # sort files which are named in the format: "video_1_subset_2.txt"

    correct = 0
    tot = 0
    for pred in pred_files:
        with open(pred, 'r') as f:
            for line in f:
                gt_ans, pred_ans = line.strip().split('\t')

                if gt_ans.endswith(")"):
                    gt_ans = gt_ans[:-1]

                if pred_ans.endswith(")"):
                    pred_ans = pred_ans[:-1]

                if len(pred_ans) == 1 and gt_ans == pred_ans:
                    correct += 1
                tot += 1

    print(f"Accuracy: {correct/tot*100:.2f}%")
    print(f"Correct: {correct}, Total: {tot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the accuracy of LMM on VQA data, given its predictions and the ground truth data")
    parser.add_argument("path", type=str, help="Path to the directory containing all the predictions files for a particular dataset.")
    args = parser.parse_args()
    main(args.path)