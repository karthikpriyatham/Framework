import json
import sys
import time
import datetime 
import framework as fw
import plotly.plotly as py

if __name__ == '__main__':
    cmd_qs, ctxs, gpus, cpus = fw.host_initialize(4, 2)
    info_file = sys.argv[1]
    info = json.loads(open(info_file).read())
    dataset = 64
    if len(sys.argv) == 4:
        dataset = int(sys.argv[3])
    if len(sys.argv) >= 3:
        partition = int(sys.argv[2])
        kernel = fw.Kernel(info, dataset=dataset, partition=partition)
    else:
        kernel = fw.Kernel(info, dataset=dataset)
        partition = info['partition']
    sched_start_time = datetime.datetime.now()
    kernel.build_kernel(gpus, cpus, ctxs)
    kernel.random_data()
    start_time, done_events = kernel.dispatch(0, 0, ctxs, cmd_qs)
    for event in done_events:
        event.wait()
    sched_end_time = datetime.datetime.now()
    seconds = (sched_end_time - sched_start_time).total_seconds()
    print("%s with partition %d and dataset %d ran %fs" % (info['name'], partition, dataset, seconds))
    try:
        cmd_qs['gpu'][0].finish()
        cmd_qs['cpu'][0].finish()
    except RuntimeError, e:
        raise
    except Exception, e:
        pass
    time.sleep(2)
    fig = fw.gantt_chart([kernel])
    from pprint import pprint
    pprint(kernel.events)
    import warnings
    warnings.filterwarnings("ignore")
    py.image.save_as(fig, '%s_%d_%d.png' % (info['name'], partition, dataset))
