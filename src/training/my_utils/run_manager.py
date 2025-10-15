import argparse
import os

import numpy as np
import pandas as pd

from training.params import parse_args as get_args_parser


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


if os.path.exists("/project/project_465000727"):
    LOGS_ROOT_PATH = '/scratch/project_465000727/logs/log_cclip'
    WANDB_API_KEY_PATH = '/project/project_465000727/wandb_api_key'
    DEFAULT_ARGS_SHEET_NAME = 'default_args_lumi'
    MMSEG_ARGS_SHEET_NAME = 'args_mmseg'
    MMDET_ARGS_SHEET_NAME = 'args_mmdet'
else:
    WANDB_API_KEY_PATH = None
    LOGS_ROOT_PATH = "/home/tim/Downloads"
    DEFAULT_ARGS_SHEET_NAME = 'default_args_meluxina'
    print('Declare LOGS_ROOT_PATH for your cluster')

SHEET_ID = '1JLfLDgJd7DTVYMyNdL9Y6CiDJBzyQBThBpqk50mMHcY'


# =========================================================================


def add_mmseg_arguments(parser):
    parser.add_argument("--mmseg_config_name", default="", type=str)
    parser.add_argument("--mmseg_work_dir", default="", type=str)
    parser.add_argument("--mmseg_frozen_stages", default=12, type=int)
    parser.add_argument("--mmseg_samples_per_gpu", default=4, type=int)
    parser.add_argument("--mmseg_lr", default=0.01, type=float)
    parser.add_argument("--mmseg_wd", default=0.01, type=float)
    parser.add_argument("--mmseg_workers_per_gpu", default=12, type=int)
    parser.add_argument("--mmseg_ntokens", default=4, type=int)
    return parser


def add_mmdet_arguments(parser):
    parser.add_argument("--mmdet_config_name", default="", type=str)
    parser.add_argument("--mmdet_work_dir", default="", type=str)
    parser.add_argument("--mmdet_frozen_stages", default=12, type=int)
    parser.add_argument("--mmdet_samples_per_gpu", default=4, type=int)
    parser.add_argument("--mmdet_lr", default=0.01, type=float)
    parser.add_argument("--mmdet_workers_per_gpu", default=12, type=int)
    return parser


def get_parser(add_mmseg_args=False, add_mmdet_args=False):
    parser = argparse.ArgumentParser("Submitit for SamNo", parents=[get_args_parser(None)])
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=60, type=int, help="Duration of the job")
    parser.add_argument("--cpu_per_task", default=12, type=int, help="Number of CPUs per GPU.")
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    # parser.add_argument('--run_eval', type=utils.bool_flag, default=True,
    #                     help="""Whether or not to also run the evaluations after training.""")
    parser.add_argument('--run_training', type=bool_flag, default=True,
                        help="""Whether or not to run the training.""")
    parser.add_argument('--repos_path', type=str, default="/home/stegmuel/projects/",
                        help="""Path where the repositories are cloned.""")
    parser.add_argument('--afterany', default="", type=str,
                        help='Job id of job that has to be finished before starting execution.')
    parser.add_argument('--od_datasets_path', default="", type=str, help='Dataset path for the OD datasets.')
    parser.add_argument('--slurm_partition', default="", type=str, help='Slurm partition.')
    parser.add_argument('--slurm_account', default="", type=str, help='Slurm account.')
    parser.add_argument('--slurm_qos', default=None, type=str, help='QOS for slurm job.')
    parser.add_argument("--gb_ram_per_gpu", default=62, type=int, help="Amount of main memory per GPU in GB")

    parser.add_argument('--python_exe', default=None, type=str,
                        help='Python bin path (e.g. from a Singularity container). Leave None to use the local Python')
    parser.add_argument('--slurm_exclusive', default=None, type=str,
                        help='Whether to use exclusive jobs or not in slurm.')
    parser.add_argument('--submitit_path_lines', default=None, type=str)
    parser.add_argument('--run_handler_process', type=bool_flag, default=True,
                        help="""Run handler process for error catching or not.""")
    parser.add_argument('--slurm_exclude', default=None, type=str,
                        help='Comma separated list of nodes to exclude.')
    parser.add_argument('--slurm_reservation', default=None, type=str,
                        help='Reservation for slurm job.')
    parser.add_argument('--slurm_hint', default=None, type=str,
                        help='Used to pass --hint=multithread when running on machine with >1 thread per core.')
    parser.add_argument("--kill_minute_threshold", default=120, type=int,
                        help="Time in minute to wait before killing job that has not changed checkpoint file.")

    parser.add_argument('--output_dir', default="", type=str,
                        help='Make code backwards compatible.')

    if add_mmseg_args:
        parser = add_mmseg_arguments(parser)
    if add_mmdet_args:
        parser = add_mmdet_arguments(parser)
    return parser


def get_full_df():
    url = "https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}".format(
        sheet_id=SHEET_ID, sheet_name='args')
    print(url)
    return pd.read_csv(url, dtype=str)


# TODO: remove url before uploading
def get_variable_args_df(logs_root_path=LOGS_ROOT_PATH):
    # read from google sheets
    df = get_full_df()
    to_run_df = df[df.run == '0'].copy()
    to_run_df['logs'] = logs_root_path
    to_run_df['output_dir'] = logs_root_path + '/' + to_run_df['name']
    to_run_df = to_run_df[[c for c in to_run_df.columns if c not in ["run"]]]
    to_run_df = to_run_df.fillna(np.nan).replace([np.nan], [None])
    return to_run_df


def get_mmseg_args_df():
    url = "https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}".format(
        sheet_id=SHEET_ID, sheet_name=MMSEG_ARGS_SHEET_NAME)
    print(url)
    df = pd.read_csv(url, dtype=str)
    to_run_df = df[df.mmseg_run == '0'].copy()
    to_run_df = to_run_df[[c for c in to_run_df.columns if c not in ["mmseg_run"]]]
    to_run_df = to_run_df.fillna(np.nan).replace([np.nan], [None])
    return to_run_df


def get_mmdet_args_df():
    url = "https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}".format(
        sheet_id=SHEET_ID, sheet_name=MMDET_ARGS_SHEET_NAME)
    print(url)
    df = pd.read_csv(url, dtype=str)
    to_run_df = df[df.mmdet_run == '0'].copy()
    to_run_df = to_run_df[[c for c in to_run_df.columns if c not in ["mmdet_run"]]]
    to_run_df = to_run_df.fillna(np.nan).replace([np.nan], [None])
    return to_run_df


def get_default_args_dict():
    url = "https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}".format(
        sheet_id=SHEET_ID, sheet_name=DEFAULT_ARGS_SHEET_NAME)
    print(url)
    return pd.read_csv(url, dtype=str).fillna(np.nan).replace([np.nan], [None]).to_dict('records')[0]


def get_all_args_lst(logs_root_path=LOGS_ROOT_PATH):
    to_run_df = get_variable_args_df(logs_root_path=logs_root_path)
    default_args_dict = get_default_args_dict()

    # Select the rows requiring to run mmseg
    # with_mmseg_df = to_run_df[to_run_df['run_mmseg'] == 'True']
    # with_mmdet_df = to_run_df[to_run_df['run_mmdet'] == 'True']
    # # without_mmstuff_df = to_run_df[(to_run_df['run_mmseg'] == 'False') & (to_run_df['run_mmdet'] == 'False')]
    # without_mmstuff_df = to_run_df

    # # Get the parser
    parser = get_parser()

    # # Get the mmseg specific args
    # mmseg_args_df = get_mmseg_args_df()

    # # Get the mmdet specific args
    # mmdet_args_df = get_mmdet_args_df()

    # # Merge the dfs
    # with_mmseg_df = with_mmseg_df.merge(mmseg_args_df, how='cross')
    # with_mmdet_df = with_mmdet_df.merge(mmdet_args_df, how='cross')

    # # Avoid duplicate work
    # run_columns = [c for c in with_mmseg_df.columns if c.startswith('run_') and c != 'run_mmseg']
    # with_mmseg_df[run_columns] = 'False'
    # run_columns = [c for c in with_mmdet_df.columns if c.startswith('run_') and c != 'run_mmdet']
    # with_mmdet_df[run_columns] = 'False'
    # without_mmstuff_df[['run_mmdet', 'run_mmseg']] = 'False'

    # run_columns = [c for c in without_mmstuff_df.columns if c.startswith('run_') and c != 'run_handler_process']
    # without_mmstuff_df = without_mmstuff_df[~(without_mmstuff_df[run_columns] == 'False').all(axis=1)]

    out_lst = []
    # loop over list
    for df in [to_run_df]:
        # for d in to_run_df.to_dict('records'):
        for d in df.to_dict('records'):
            tmp = []
            # loop over default arguments
            for k, v in default_args_dict.items():
                if v is not None:
                    tmp.append('--{}'.format(k))
                    if k == 'python_exe':
                        tmp.append('{}'.format(v))
                    else:
                        for i in v.split():
                            tmp.append('{}'.format(i))
                elif k in ['lock_image', 'pretrained_image', 'lock-image', 'pretrained-image', 'save-most-recent']:
                    tmp.append('--{}'.format(k))

            # loop over variable arguments
            for k, v in d.items():
                if v is not None:
                    tmp.append('--{}'.format(k))
                    if k == 'python_exe':
                        tmp.append('{}'.format(v))
                    else:
                        for i in v.split():
                            tmp.append('{}'.format(i))
                elif k in ['gather-with-grad', 'local-loss']:
                    tmp.append('--{}'.format(k))
            out_lst.append(parser.parse_args(tmp))
    return out_lst
