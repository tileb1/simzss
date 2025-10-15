import json
import os
import re
import sys
from collections import defaultdict

import wandb


###


def update_wandb_run_with_raw_metadata(NAME, metadata, args):
    RAW = {NAME: metadata}
    update_wandb_run(NAME, process_struct(RAW)[NAME], args)


def update_wandb_run(NAME, metadata, args):
    wandb.init(
        entity="ssl-goodpapers",
        project=args.wandb_project_name,
        name=NAME,
        id=NAME,
        resume=True,
        mode=args.wandb_mode,
    )

    # Make interactive plot (can bug sometimes)
    for index in range(len(metadata[next(iter(metadata.keys()))])):
        to_log = {}
        for k, v in metadata.items():
            epoch, miou = v[index]
            to_log['val/epoch'] = epoch
            to_log['val/' + k] = miou
        print(to_log)
        wandb.log(to_log)

    # Make non-interactive plot
    to_log = {}
    for k, v in metadata.items():
        if k.startswith('_'):
            continue
        to_log['lineplot_' + k] = wandb.plot.line(wandb.Table(data=v, columns=["val/epoch", "mIoU"]), "val/epoch",
                                                  "mIoU", title=k)
    wandb.log(to_log)

    # Finish
    wandb.finish()


def process_struct(EVAL_STRUCT):
    EVAL_STRUCT_OUT = {}
    MAX_DICT_VAL = defaultdict(float)
    MAX_DICT_KEY = {}

    for k, v in EVAL_STRUCT.items():
        inner = defaultdict(list)
        if 'epoch_latest.pt' in v:
            del v['epoch_latest.pt']
        print(v.keys())
        for kk in sorted(v.keys(), key=lambda x: int(re.sub("[^0-9]", "", x))):
            epoch = int(re.sub("[^0-9]", "", kk))
            for dataset in v[kk].keys():
                if isinstance(v[kk][dataset], dict):
                    print(v[kk][dataset].keys())
                    for threshold_str in sorted(v[kk][dataset].keys(), key=lambda x: float(x)):
                        inner[dataset + '_' + threshold_str + '_bkg'].append((epoch, v[kk][dataset][threshold_str]))
                        if v[kk][dataset][threshold_str] > MAX_DICT_VAL[dataset]:
                            MAX_DICT_VAL[dataset] = v[kk][dataset][threshold_str]
                            MAX_DICT_KEY[dataset] = dataset + '_' + threshold_str + '_bkg'
                else:
                    inner[dataset].append((epoch, v[kk][dataset]))
        EVAL_STRUCT_OUT[k] = dict(inner)

    EVAL_STRUCT_OUT_NEW = {}
    for kk in EVAL_STRUCT_OUT:
        inside = EVAL_STRUCT_OUT[kk]
        EVAL_STRUCT_OUT_NEW[kk] = {k: inside[k] for k in inside.keys() if
                                   not '_bkg' in k or ('_bkg' in k and k in MAX_DICT_KEY.values())}
    return EVAL_STRUCT_OUT_NEW


if __name__ == "__main__":
    os.chdir('/scratch/project_465000727/repos/Contextual-CLIP')
    src = '/scratch/project_465000727/repos/Contextual-CLIP/src'
    if src not in sys.path:
        sys.path = [src] + sys.path

    training = '/scratch/project_465000727/repos/Contextual-CLIP/src/training'
    if training not in sys.path:
        sys.path.append(training)

    from my_utils.run_manager import get_all_args_lst

    args = get_all_args_lst()[0]

    ROOT = '/scratch/project_465000727/logs/log_cclip/'

    EVAL_STRUCT = {}


    def condition(i):
        # cond = re.match('25044[12]_[78].*', i)
        cond = i.startswith('240503_bconly')
        return cond


    for i in [i for i in os.listdir(ROOT) if condition(i)]:
        try:
            with open(os.path.join(ROOT, i, 'checkpoints/results_custom_eval.json'), 'r') as f:
                EVAL_STRUCT[i] = json.load(f)
            print('GOOD:', i)
        except:
            print('NO DATA:', i)

    # EVAL_STRUCT_OUT = {}
    # for k, v in EVAL_STRUCT.items():
    #     inner = defaultdict(list)
    #     for kk in sorted(v.keys(), key=lambda x : int(re.sub("[^0-9]", "", x))):
    #         for dataset in v[kk].keys():
    #             inner[dataset].append((int(re.sub("[^0-9]", "", kk)), v[kk][dataset]))
    #     EVAL_STRUCT_OUT[k] = dict(inner)

    EVAL_STRUCT_OUT = process_struct(EVAL_STRUCT)

    for NAME, metadata in EVAL_STRUCT_OUT.items():
        update_wandb_run(NAME, metadata, args)
