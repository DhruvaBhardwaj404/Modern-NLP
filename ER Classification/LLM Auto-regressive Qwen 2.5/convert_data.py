import json
import os
import sys
def convert_labels(train, map, new_train):
    with open(map, 'r', encoding='utf-8') as f:
        label_map = json.load(f)

    output_lines = []
    with open(train, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                if "relationMentions" in data:
                    for mention in data["relationMentions"]:
                        old_label = mention.get("label")
                        if old_label in label_map:
                            mention["label"] = label_map[old_label]

                output_lines.append(json.dumps(data, ensure_ascii=False))

            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line[:50]}...")

    with open(new_train, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + '\n')

def convert_json_to_jsonl(input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in data:
                json_record = json.dumps(entry, ensure_ascii=False)
                f.write(json_record + '\n')

def convert_to_valid_json(input_filepath, output_filepath):
    """
    Reads a file containing concatenated/stacked multi-line JSON objects
    and saves them as a single valid JSON array.
    """
    with open(input_filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    records = []
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(content):
        # 1. Skip whitespaces, newlines, or stray commas between objects
        while idx < len(content) and content[idx] in ' \t\n\r,':
            idx += 1

        if idx >= len(content):
            break

        # 2. Extract the next complete JSON object
        try:
            obj, end_idx = decoder.raw_decode(content, idx)
            records.append(obj)
            idx = end_idx  # Move the index forward to the end of the parsed object
        except json.JSONDecodeError as e:
            print(f"Failed to decode at character index {idx}: {e}")
            break
    # 3. Write the extracted records into a new file as a proper JSON array
    with open(output_filepath, 'w', encoding='utf-8') as f_out:
        # ensure_ascii=False keeps the Hindi characters intact instead of escaping them
        json.dump(records, f_out, ensure_ascii=False, indent=4)
    print(f"Success! Converted {len(records)} objects into a valid JSON array at '{output_filepath}'.")


if __name__ == "__main__":
    output_dir = sys.argv[1]
    files = ["tcy_val.jsonl","or_train.jsonl"]
    names = ["tcy","or"]
    path = "../sft_dataset"

    os.makedirs(output_dir, exist_ok=True)

    for file, name in zip(files,names):
        convert_to_valid_json(os.path.join(path,file),os.path.join(output_dir,f"{name}.json"))
        convert_json_to_jsonl(os.path.join(output_dir,f"{name}.json"),os.path.join(output_dir,f"{name}_train.jsonl"))
        convert_labels(os.path.join(output_dir,f"{name}_train.jsonl"),os.path.join(path,f"{name}_map.json"),os.path.join(output_dir,f"{name}_train.jsonl"))

    print("converted!")