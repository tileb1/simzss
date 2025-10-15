import os
import pickle
import random
import shutil
import socket
import subprocess
import sys
import time
from collections import defaultdict

from training.my_utils.run_manager import LOGS_ROOT_PATH


def signal_handler(args, sig, frame):
    pass
    # print(args)
    # clean_on_leave(args)
    # print('clean_on_leave() was called')


def pin_workers_iterator(the_iterator, args):
    if 'karolina' in socket.gethostname() or 'ristoteles' in socket.gethostname():
        return
    try:
        print(args.cpus)
    except AttributeError:
        args.cpus = list(sorted(os.sched_getaffinity(0)))
        # os.system("taskset -p -c %d %d" % ((args.cpus[0]), os.getpid()))

    if args.num_workers > 0:
        for index, w in enumerate(the_iterator._workers):
            os.system("taskset -p -c %d %d" % ((args.cpus[(index + 1) % len(args.cpus)]), w.pid))


def clean_on_leave(args):
    if args.untar_path[:8] == '/dev/shm' and int(args.gpu) == 0:
        shutil.rmtree(args.untar_path)


def singularity_x_slurm():
    if os.path.exists("/project/project_465000727") and os.environ['USER'] == 'tilebail':
        # with open('/etc/passwd', 'r') as f:
        #     out = f.readlines()
        # for l in out:
        #     if 'slurm:x:982:982::/home/slurm:/sbin/nologin' in l:
        #         return
        # subprocess.run('echo "slurm:x:982:982::/home/slurm:/sbin/nologin" >> /etc/passwd', shell=True)
        # subprocess.run('echo "slurm:x:982:" >> /etc/group', shell=True)
        # os.environ["LD_LIBRARY_PATH"] = "/usr/lib64:" + os.environ["LD_LIBRARY_PATH"]
        os.environ["PATH"] = "/project/project_465000727/hostbin:" + os.environ["PATH"]
    print('PATH:', os.environ["PATH"])
    print('LD_LIBRARY_PATH:', os.environ["LD_LIBRARY_PATH"])


def clear_shm():
    folder = '/dev/shm'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            pass


def check_checkpointing_freq(args, time_ok, elapsed_time):
    path = os.path.join(args.output_dir, 'checkpoint.pth')
    if os.path.exists(path) and elapsed_time > args.kill_minute_threshold * 60:
        time_ok = os.path.getmtime(path)
    # allow for 30 mins to create checkpoint (should take normally about 10 mins)
    if int(time.time()) - time_ok > args.kill_minute_threshold * 60 and 'output_knn' not in os.listdir(args.output_dir):
        # kill job
        return_code = 1
        while return_code == 1:
            out = subprocess.run("scancel $SLURM_JOB_ID", shell=True, env=dict(os.environ))
            return_code = out.returncode
            time.sleep(60)
        sys.exit(1)
    return time_ok


def handler_process(args, main):
    if os.path.exists("/project/project_465000727") and args.run_handler_process:
        args_lst = get_args_lst(args)
        dict_count = defaultdict(int)
        with open(os.path.join(LOGS_ROOT_PATH, args.output_dir, 'handler_process.txt'), 'a+') as f:
            f.write(str(args_lst) + '\n')

        time_ok = int(time.time())
        start_time = int(time.time())
        while True:
            try:
                # release_held_jobs()
                # time_ok = check_checkpointing_freq(args, time_ok, int(time.time()-start_time))
                tmp = subprocess.run(
                    'squeue --me --format="%.18i %.400j %.8u %.2t %.10M %.6D %R" --sort -M,j | tail -n +2', shell=True,
                    env=dict(os.environ), capture_output=True)
                running_jobs_raw = tmp.stdout.decode("utf-8")

                # bad_nodes = get_exclude_list(args)
                respawn_failed_jobs(args_lst, running_jobs_raw, dict_count, main, args, '')

                with open(os.path.join(LOGS_ROOT_PATH, args.output_dir, 'handler_process.txt'), 'a+') as f:
                    f.write(str(dict_count) + '\n')
                time.sleep(120 + random.randint(0, 120))
            except RuntimeError as e:
                with open(os.path.join(LOGS_ROOT_PATH, args.output_dir, 'handler_process.txt'), 'a+') as f:
                    f.write(str(e) + '\n')
                time.sleep(300 + random.randint(0, 120))


def respawn_failed_jobs(args_lst, running_jobs_raw, dict_count, main, args_me, bad_nodes):
    lst_raw = running_jobs_raw.strip('\n').split('\n')
    lst_job_name = [i.split()[1] for i in lst_raw]
    lst_state = [i.split()[-4] for i in lst_raw]

    for args in args_lst:
        dict_count[args.output_dir] += 1
        for job_name, s in zip(lst_job_name, lst_state):
            if args.output_dir.split('/')[-1] == job_name and s != 'CG':
                # This job is running correctly
                dict_count[args.output_dir] = 0
                break

    lst_running_job_name = [i.split()[1] for i in lst_raw if
                            "BeginTime" not in i and "launch failed requeued held" not in i and 'nid' in i]
    lst_submitted_alongside = [a.output_dir.split('/')[-1] for a in args_lst]
    lst_running_job_name = list(sorted(list(set(lst_running_job_name).intersection(set(lst_submitted_alongside)))))

    for a in args_lst:
        if os.path.exists(os.path.join(a.output_dir, 'output_knn')):
            dict_count[a.output_dir] = 0
            continue
        # log_path = os.path.join(a.output_dir, "output_linear", "log.txt")
        # if os.path.exists(log_path):
        #     with open(log_path, 'r') as f:
        #         out = f.readlines()[-1]
        #         loaded = json.loads(out)
        #     if loaded['epoch'] == 99:
        #         dict_count[a.output_dir] = 0
        #         continue
        elif dict_count[a.output_dir] >= 2 and args_me.output_dir.split('/')[-1] == lst_running_job_name[0]:
            # Only 1 running job should reschedule
            main(a, bad_nodes=bad_nodes)
            dict_count[a.output_dir] = 0
        elif dict_count[a.output_dir] >= 4 and args_me.output_dir.split('/')[-1] == lst_running_job_name[1]:
            # Only 1 running job should reschedule
            main(a, bad_nodes=bad_nodes)
            dict_count[a.output_dir] = 0
        elif dict_count[a.output_dir] >= 6 and args_me.output_dir.split('/')[-1] == lst_running_job_name[2]:
            # Only 1 running job should reschedule
            main(a, bad_nodes=bad_nodes)
            dict_count[a.output_dir] = 0
        elif dict_count[a.output_dir] >= 8 and args_me.output_dir.split('/')[-1] == lst_running_job_name[3]:
            # Only 1 running job should reschedule
            main(a, bad_nodes=bad_nodes)
            dict_count[a.output_dir] = 0


def get_args_lst(args):
    folder = os.path.join(LOGS_ROOT_PATH, 'args_lst')
    current = {'time': 0, 'args_lst': []}
    for f in os.listdir(folder):
        with open(os.path.join(folder, f), 'rb') as handle:
            out = pickle.load(handle)
            for a in out['args_lst']:
                if a.output_dir == args.output_dir and current['time'] < out['time']:
                    current = out
    return current['args_lst']
