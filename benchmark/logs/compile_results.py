import glob
import os
import re
import json
import sys
from collections import defaultdict, OrderedDict

# Root directory to search (defaults to the folder holding this script)
root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))

# Optional filter: only process runs of this size. Log files are named
# syncer-n_{n}_{num_messages}_{batch_size}_{compr_factor}.log, so this is matched
# against the num_messages field rather than anywhere in the name.
num_messages = None
if len(sys.argv) > 2:
    try:
        num_messages = int(sys.argv[2])
    except ValueError:
        print(f"Usage: {os.path.basename(__file__)} [root] [num_messages]")
        sys.exit(1)

# Regex to pull the run parameters out of a syncer log file name
name_pattern = re.compile(r'^syncer-n_(\d+)_(\d+)_(\d+)_(\d+)\.log$')

# Regex to extract latency array and value from line
line_pattern = re.compile(r'with latency \[([^\]]+)\], status \{"([^"]+)"\}')

# Regex to extract the shuffled output array that is logged alongside the latency
value_pattern = re.compile(r'and value \{(.*)\}\s*$')


def count_elements(value_str):
    """Number of entries in a Rust Debug-printed array of strings.

    The entries are raw bytes, so they can contain '[', ',' and escaped quotes;
    splitting on those miscounts. Walk the string instead and count the quoted
    literals, skipping any character that follows a backslash.
    """
    count = 0
    index = 0
    in_string = False
    while index < len(value_str):
        char = value_str[index]
        if in_string:
            if char == '\\':
                index += 2
                continue
            if char == '"':
                in_string = False
                count += 1
        elif char == '"':
            in_string = True
        index += 1
    return count


# Ordered dictionary to maintain insertion order
latency_by_category = OrderedDict()

# Length of the output array found in each log file
output_lengths_by_file = OrderedDict()

# Process all matching log files, including those in subfolders
logfiles = sorted(glob.glob(os.path.join(root, "**", "syncer-*.log"), recursive=True))

if num_messages is not None:
    matching = []
    for filepath in logfiles:
        name_match = name_pattern.match(os.path.basename(filepath))
        if name_match and int(name_match.group(2)) == num_messages:
            matching.append(filepath)
    skipped = len(logfiles) - len(matching)
    logfiles = matching
    if skipped:
        print(f"Skipping {skipped} log file(s) not run with num_messages={num_messages}")

if not logfiles:
    if num_messages is not None:
        print(f"No syncer-*.log files with num_messages={num_messages} found under {root}")
    else:
        print(f"No syncer-*.log files found under {root}")
    sys.exit(1)

print(f"Found {len(logfiles)} log file(s) under {root}:")
for filepath in logfiles:
    print(f"  {os.path.relpath(filepath, root)}")

for filepath in logfiles:
    output_lengths = OrderedDict()
    with open(filepath, 'r') as file:
        for line in file:
            match = line_pattern.search(line)
            if match:
                latency_str, category = match.groups()
                latency_array = [int(x.strip()) for x in latency_str.split(',')]
                if category not in latency_by_category:
                    latency_by_category[category] = []
                latency_by_category[category].extend(latency_array)

                value_match = value_pattern.search(line)
                if value_match:
                    length = count_elements(value_match.group(1))
                    if length:
                        output_lengths[category] = length
    output_lengths_by_file[os.path.relpath(filepath, root)] = output_lengths

# Print output array lengths
print("\nOutput array lengths:")
for filename, lengths in output_lengths_by_file.items():
    if lengths:
        for category, length in lengths.items():
            print(f"  {filename}: {category} -> {length} elements")
    else:
        print(f"  {filename}: no output array found")

# Compute average latencies
average_latencies = OrderedDict()
for category, latencies in latency_by_category.items():
    if latencies:
        avg = sum(latencies) / len(latencies)
        average_latencies[category] = avg
    else:
        average_latencies[category] = None

# Print average latencies
print("\nAverage latencies per category:")
for category, avg in average_latencies.items():
    print(f"  {category}: {avg:.2f} ms")

# Compute and print latency differences
print("\nLatency differences between subsequent categories:")
previous_category = None
previous_avg = None
for category, avg in average_latencies.items():
    if previous_category is not None:
        diff = avg - previous_avg
        print(f"  {previous_category} → {category}: {diff:.2f} ms")
    previous_category, previous_avg = category, avg
