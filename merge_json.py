import json

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

# Example usage
input_file = 'multi_agent_logs/events_2025-06-06.json'  # replace with your input file
output_file = 'multi_agent_logs/merged_json.json'
merge_json_lines(input_file, output_file)
