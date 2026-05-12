"""
Offline analysis script for NoitaRL.
Reads data/episode_history.csv and actions_trace.jsonl to generate statistics and plots.
"""

import os
import json
import csv
import matplotlib.pyplot as plt
from collections import Counter

def analyze_actions():
    trace_file = "actions_trace.jsonl"
    if not os.path.exists(trace_file):
        print(f"{trace_file} not found. Skipping action analysis.")
        return

    print(f"Parsing {trace_file}...")
    action_counts = Counter()
    total_actions = 0

    action_names = {
        0: "IDLE", 1: "LEFT", 2: "RIGHT", 3: "JUMP",
        4: "L+JMP", 5: "R+JMP", 6: "FIRE", 7: "DIG_D",
    }

    try:
        with open(trace_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    act = data.get("a", 0)
                    action_counts[act] += 1
                    total_actions += 1
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"Error reading {trace_file}: {e}")
        return

    if total_actions == 0:
        print("No actions found.")
        return

    print("\n--- Action Distribution ---")
    labels = []
    sizes = []
    for act, count in sorted(action_counts.items()):
        name = action_names.get(act, str(act))
        pct = (count / total_actions) * 100
        print(f"{name:8s}: {count:8d} ({pct:5.2f}%)")
        labels.append(name)
        sizes.append(count)

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title("Action Distribution")
    plt.tight_layout()
    out_file = "action_distribution.png"
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")


def analyze_episodes():
    csv_file = "data/episode_history.csv"
    if not os.path.exists(csv_file):
        print(f"{csv_file} not found. Skipping episode analysis.")
        return

    print(f"\nParsing {csv_file}...")
    x_coords = []
    y_coords = []  # Depth

    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Treat max_x and max_depth as the proxy for death/timeout location
                    x = float(row.get("max_x", 0))
                    y = float(row.get("max_depth", 0))
                    if x != 0 or y != 0:
                        x_coords.append(x)
                        y_coords.append(y)
                except ValueError:
                    pass
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return

    if not x_coords:
        print("No valid episode data found.")
        return

    print(f"Found {len(x_coords)} episodes.")

    plt.figure(figsize=(10, 8))
    # In Noita, larger Y is deeper. So we invert the Y axis for intuitive plotting.
    plt.hexbin(x_coords, y_coords, gridsize=30, cmap="inferno", mincnt=1)
    cb = plt.colorbar(label='Count in bin')
    plt.gca().invert_yaxis()
    plt.xlabel("Max X (pixels)")
    plt.ylabel("Max Depth Y (pixels) - Inverted")
    plt.title("Episode Heatmap (Max X vs Max Depth)")
    plt.tight_layout()
    out_file = "episode_heatmap.png"
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")


if __name__ == "__main__":
    analyze_actions()
    analyze_episodes()
