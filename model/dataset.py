def load_labels(ascii_path):
    labels = {}
    
    with open(ascii_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.strip().split(" ")

            if parts[1] == "err":
                continue
        
            img_id = parts[0]

            text = parts[8].replace("|", " ")

            labels[img_id] = text
    
    return labels