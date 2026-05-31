from dataset import load_labels

def main():
    path = "data/iam/ascii/lines.txt"
    labels = load_labels(path)
    print(f"Loaded {len(labels)} labels from {path}")

if __name__ == "__main__":
	main()