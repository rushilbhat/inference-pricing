#!/usr/bin/env python3
import json
import argparse
import matplotlib.pyplot as plt
import config

COLOURS = ['#6B7280', '#3B82F6', '#10B981', '#F59E0B', '#EF4444', 
          '#8B5CF6', '#EC4899', '#14B8A6', '#F97316', '#06B6D4']

METRICS = {
    'input_tps': ('tokens/sec', False),
    'output_tps': ('tokens/sec', False),
    'p90_ttft': ('seconds', True),
    'p90_tpot': ('milliseconds', False)
}

def load_data(jsonl_path):
    data = {}
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                measurements = entry['measurements']
                data[entry['id']] = {
                    'batch_sizes': [m['batch_size'] for m in measurements],
                    'input_tps': [m['input_tps'] for m in measurements],
                    'output_tps': [m['output_tps'] for m in measurements],
                    'p90_ttft': [m['p90_ttft'] for m in measurements],
                    'p90_tpot': [m['p90_tpot'] * 1000 for m in measurements]
                }
    return data

def plot_panel(ax, data, key):
    units, log_y = METRICS[key]
    title = key.replace('_', ' ').upper()
    batch_sizes = sorted(set(bs for d in data.values() for bs in d['batch_sizes']))
    for i, (config_id, d) in enumerate(data.items()):
        ax.plot(d['batch_sizes'], d[key], linewidth=1, 
                color=COLOURS[i % len(COLOURS)], label=config_id, alpha=0.8)
        
    x_start = batch_sizes[0]
    if key in ['p90_ttft', 'p90_tpot']:
        is_tpot = key == 'p90_tpot'
        for name, ttft_limit, tpot_limit in config.SLO_LEVELS:
            limit = tpot_limit * 1000 if is_tpot else ttft_limit
            unit = 'ms' if is_tpot else 's'
            label_value = int(limit) if is_tpot else limit
            ax.axhline(y=limit, color='#555555', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.text(x_start, limit, f" {name} ({label_value}{unit})", color='#555555', fontsize=9, va='bottom', fontweight='bold')
    
    ax.set_xlabel('Batch Size', fontsize=11)
    ax.set_ylabel(f'{title} ({units})', fontsize=11)
    ax.set_title(f'{title} vs Batch Size', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    if log_y: ax.set_yscale('log')
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels(batch_sizes)
    ax.legend(loc='best', fontsize=10)

def create_plots(data, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Batch Size Performance Analysis', fontsize=16, fontweight='bold')
    
    for ax, key in zip(axes.flat, METRICS.keys()):
        plot_panel(ax, data, key)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('jsonl_file')
    args = parser.parse_args()
    
    input_path = args.jsonl_file
    output_path = input_path.rsplit('.', 1)[0] + '_plot.png'
    
    data = load_data(input_path)
    create_plots(data, output_path)
    print(f"Saved: {output_path}")