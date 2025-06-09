import json
import argparse


def merge_json_lines(input_file_path, output_file_path):
    merged_data = []

    with open(input_file_path, 'r') as infile:
        for line in infile:
            line = line.strip()
            if line:  # skip empty lines
                try:
                    json_obj = json.loads(line)
                    merged_data.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {e}")

    with open(output_file_path, 'w') as outfile:
        json.dump(merged_data, outfile, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge JSON lines into a single JSON array.')
    parser.add_argument('input_file', help='Path to the input JSON lines file')
    parser.add_argument('output_file', help='Path to the output merged JSON file')
    args = parser.parse_args()

    merge_json_lines(args.input_file, args.output_file)
# Example usage
#input_file = 'multi_agent_logs/events_2025-06-08.json'  # replace with your input file
#output_file = 'multi_agent_logs/merged_json.json'
#merge_json_lines(input_file, output_file)
