"""Utilities for visualising model schedules as Gantt charts."""

import argparse
import csv
import os
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from common_utils import greedy_select_action, setup_seed
from data_utils import pack_data_from_config
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from model.PPO import PPO_initialize
from params import configs


def _load_instance(data_source: str, data_name: str, instance_index: int) -> Tuple:
    """Load a single FJSP instance specified by the configuration."""

    data_entries = pack_data_from_config(data_source, [data_name])
    if not data_entries:
        raise FileNotFoundError(f"No data found for {data_source}/{data_name}.")

    (job_lengths, op_pts), _ = data_entries[0]
    if instance_index < 0 or instance_index >= len(job_lengths):
        raise IndexError(
            f"Instance index {instance_index} is out of range for dataset size {len(job_lengths)}."
        )

    return job_lengths[instance_index], op_pts[instance_index]


def _run_greedy_schedule(job_length, op_pt, model_path: str, seed: int) -> Tuple[List[Dict], float]:
    """Run the greedy policy on a single instance and record the schedule log."""

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
    device = torch.device(configs.device)

    setup_seed(seed)

    ppo = PPO_initialize()
    state_dict = torch.load(model_path, map_location=device)
    ppo.policy.load_state_dict(state_dict)
    ppo.policy.to(device)
    ppo.policy.eval()

    n_j = job_length.shape[0]
    n_op, n_m = op_pt.shape
    env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)

    state = env.set_initial_data([job_length], [op_pt])

    while True:
        with torch.no_grad():
            pi, _ = ppo.policy(fea_j=state.fea_j_tensor,
                               op_mask=state.op_mask_tensor,
                               candidate=state.candidate_tensor,
                               fea_m=state.fea_m_tensor,
                               mch_mask=state.mch_mask_tensor,
                               comp_idx=state.comp_idx_tensor,
                               dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                               fea_pairs=state.fea_pairs_tensor)

        action = greedy_select_action(pi)
        state, _, done = env.step(actions=action.cpu().numpy())
        if done:
            break

    schedule_log = env.get_schedule_log(0)
    makespan = env.current_makespan[0]
    return schedule_log, makespan


def _plot_gantt(schedule_log: List[Dict], num_jobs: int, num_machines: int, title: str):
    """Create a matplotlib figure representing the schedule as a Gantt chart."""

    fig_height = max(4, 0.8 * num_machines + 2)
    fig_width = max(8, 0.8 * num_jobs + 6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = plt.get_cmap('tab20', max(num_jobs, 1))

    legend_handles = [
        mpatches.Patch(color=cmap(job_idx % cmap.N), label=f'Job {job_idx + 1}')
        for job_idx in range(num_jobs)
    ]

    for machine_idx in range(num_machines):
        machine_tasks = [entry for entry in schedule_log if entry['machine'] == machine_idx]
        if not machine_tasks:
            continue

        task_spans = []
        colors = []
        for task in machine_tasks:
            start = task['start_time']
            duration = task['end_time'] - task['start_time']
            task_spans.append((start, duration))
            colors.append(cmap(task['job'] % cmap.N))

        ax.broken_barh(task_spans, (machine_idx - 0.4, 0.8), facecolors=colors, edgecolors='black')

        for span, task in zip(task_spans, machine_tasks):
            ax.text(span[0] + span[1] / 2,
                    machine_idx,
                    f"J{task['job'] + 1}-O{task['operation'] + 1}",
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='black')

    ax.set_ylim(-1, num_machines)
    ax.set_xlim(0, max(entry['end_time'] for entry in schedule_log) * 1.05 if schedule_log else 1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_title(title)
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f'M{idx + 1}' for idx in range(num_machines)])
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    if legend_handles:
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0.)

    fig.tight_layout()
    return fig


def _export_schedule(schedule_log: List[Dict], csv_path: str):
    """Export the recorded schedule to CSV for downstream analysis."""

    directory = os.path.dirname(csv_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(csv_path, 'w', newline='') as csv_file:
        fieldnames = ['job', 'operation', 'global_operation', 'machine',
                      'start_time', 'end_time', 'processing_time']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in schedule_log:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualise DANIEL schedule as a Gantt chart.')
    parser.add_argument('--model-source', default=configs.model_source, help='Directory key under trained_network/.')
    parser.add_argument('--model-name', default=configs.test_model[0], help='Model checkpoint name without extension.')
    parser.add_argument('--data-source', default=configs.data_source, help='Data source directory under data/.')
    parser.add_argument('--data-name', default=configs.test_data[0], help='Dataset folder to load instances from.')
    parser.add_argument('--instance-index', type=int, default=0, help='Index of the instance within the dataset folder.')
    parser.add_argument('--seed', type=int, default=configs.seed_test, help='Random seed for reproducibility.')
    parser.add_argument('--save-path', default=None, help='Optional path to save the Gantt chart image.')
    parser.add_argument('--csv-path', default=None, help='Optional path to export the raw schedule as CSV.')
    parser.add_argument('--dpi', type=int, default=200, help='DPI for the saved figure.')
    parser.add_argument('--show', action='store_true', help='Display the chart window in addition to saving.')
    return parser.parse_args()


def main():
    args = parse_args()

    job_length, op_pt = _load_instance(args.data_source, args.data_name, args.instance_index)

    model_path = os.path.join('./trained_network', args.model_source, f'{args.model_name}.pth')
    schedule_log, makespan = _run_greedy_schedule(job_length, op_pt, model_path, args.seed)

    num_jobs = job_length.shape[0]
    num_machines = op_pt.shape[1]
    title = (f'Gantt Chart | Model: {args.model_name} | Data: {args.data_name} '
             f'| Instance #{args.instance_index + 1} | Makespan: {makespan:.2f}')

    fig = _plot_gantt(schedule_log, num_jobs, num_machines, title)

    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(args.save_path, dpi=args.dpi)
        print(f'Saved Gantt chart to {args.save_path}')
    else:
        default_dir = './gantt_charts'
        os.makedirs(default_dir, exist_ok=True)
        default_path = os.path.join(
            default_dir,
            f'{args.model_name}_{args.data_name}_idx{args.instance_index + 1}.png'
        )
        fig.savefig(default_path, dpi=args.dpi)
        print(f'Saved Gantt chart to {default_path}')

    if args.csv_path:
        _export_schedule(schedule_log, args.csv_path)
        print(f'Saved schedule log to {args.csv_path}')

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()

