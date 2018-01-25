#!/usr/bin/env python

# from fw import *
import json, sys, datetime, time, heapq
import numpy as np
import pyschedcl as fw
import logging
import argparse

import resource, sys
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)


def parse_arg(args=None):
    parser = argparse.ArgumentParser(description='Schedule set of independent OpenCL kernels on CPU-GPU heterogeneous multicores')
    parser.add_argument('-f', '--file',
                        help='Input task file containing list of <json filename, partition class, dataset> tuples',
                        required='True')
    parser.add_argument('-s', '--select',
                        help='Scheduling heuristic (baseline, lookahead, adbias) ',
                        default='baseline')
    parser.add_argument('-ng', '--nGPU',
                        help='Number of GPUs',
                        default='4')
    parser.add_argument('-nc', '--nCPU',
                        help='Number of CPUs',
                        default='2')
    parser.add_argument('-l', '--log',
                        help='Flag for turning on LOG',
                        action="store_true")
    parser.add_argument('-g', '--graph',
                        help='Flag for plotting GANTT chart for execution',
                        action="store_true")
    parser.add_argument('-df', '--dump_output_file',
                        help='Flag for dumping output file for a kernel',
                        action="store_true")
    return parser.parse_args(args)

# logging.basicConfig(level=logging.DEBUG)


def baseline_select(kernels, **kwargs):
    mixed_q = kwargs['M']
    gpu_q = kwargs['G']
    cpu_q = kwargs['C']

    now_kernel = None
    if fw.nCPU > 0 and fw.nGPU > 0 and mixed_q:
        i_p, i_num_global_work_items, i = heapq.heappop(mixed_q)
        i_num_global_work_items, i_p = kernels[i].get_num_global_work_items(), kernels[i].partition

        now_kernel = (i, i_p)
    elif fw.nCPU > 0 and cpu_q:
        c_p, c_num_global_work_items, c_i = heapq.heappop(cpu_q)
        now_kernel = (c_i, 0)
    elif fw.nGPU > 0 and gpu_q:
        g_p, g_num_global_work_items, g_i = heapq.heappop(gpu_q)
        now_kernel = (g_i, 10)
    if now_kernel is not None:
        i, p =now_kernel
        logging.debug( "Selecting kernel " + kernels[i].name + " with partition class value " + str(p))
    return now_kernel, None
    

def look_ahead_select(kernels, **kwargs):
    mixed_q = kwargs['M']
    gpu_q = kwargs['G']
    cpu_q = kwargs['C']
    now_kernel, next_kernel = None, None
    if fw.nCPU > 0 and fw.nGPU > 0 and len(mixed_q) >= 2:
        i_p, i_num_global_work_items, i = heapq.heappop(mixed_q)
        j_p, j_num_global_work_items, j = heapq.heappop(mixed_q)
        i_num_global_work_items, j_num_global_work_items = kernels[i].get_num_global_work_items(), kernels[j].get_num_global_work_items()
        i_p, j_p = kernels[i].partition, kernels[j].partition
        if i_p >= 8 and j_p <= 2:
            next_kernel = (j, 0)
            now_kernel = (i, 10)
        elif j_p >= 8 and i_p <= 2:
            next_kernel = (i, 0)
            now_kernel = (j, 10)
        elif i_num_global_work_items < j_num_global_work_items:
            now_kernel = (i, i_p)
            heapq.heappush(mixed_q, (abs(j_p-5), -j_num_global_work_items, j))
        else:
            now_kernel = (j, j_p)
            heapq.heappush(mixed_q, (abs(i_p-5), -i_num_global_work_items, i))
    elif fw.nCPU > 0 and fw.nGPU > 0 and mixed_q:
        i_p, i_num_global_work_items, i = heapq.heappop(mixed_q)
        i_num_global_work_items = kernels[i].get_num_global_work_items()
        i_p = kernels[i].partition
        now_kernel = (i, i_p)
    elif fw.nCPU > 0 and cpu_q:
        c_p, c_num_global_work_items, c_i = heapq.heappop(cpu_q)
        now_kernel = (c_i, 0)
    elif fw.nGPU > 0 and gpu_q:
        g_p, g_num_global_work_items, g_i = heapq.heappop(gpu_q)
        now_kernel = (g_i, 10)
    if now_kernel is not None:
        i, p = now_kernel
        logging.debug( "Selecting kernel " + kernels[i].name + " with partition class value " + str(p))
    return now_kernel, next_kernel    
            

def adaptive_bias_select(kernels, **kwargs):
    mixedc_q = kwargs['M1']
    mixedg_q = kwargs['M2']
    gpu_q = kwargs['G']
    cpu_q = kwargs['C']
    num_global_work_items_max =kwargs['feat_max']
    t = kwargs['threshold']
    now_kernel, next_kernel = None, None
    if fw.nCPU > 0 and fw.nGPU > 0 and mixedc_q and mixedg_q:
        is_dispatched = 0
        c_p, c_num_global_work_items, c_i = heapq.heappop(mixedc_q)
        g_p, g_num_global_work_items, g_i = heapq.heappop(mixedg_q)
        c_num_global_work_items, g_num_global_work_items = kernels[c_i].get_num_global_work_items(), kernels[g_i].get_num_global_work_items()
        c_p, g_p = kernels[c_i].partition, kernels[g_i].partition

        if c_p <= 2 and g_p >= 8:
            next_kernel = (c_i, 0)
            now_kernel = (g_i, 10)
            is_dispatched = 1

        elif c_p <= 4 and g_p >= 6:
            dispatch_cpu,  dispatch_gpu = False, False
            if c_num_global_work_items < t*num_global_work_items_max:
                now_kernel = (c_i, 0)
                is_dispatched +=1
                dispatch_cpu = True
            if g_num_global_work_items < t*num_global_work_items_max:
                if now_kernel == None:
                    now_kernel = (g_i, 10)
                else:
                    next_kernel = (g_i, 10)
                is_dispatched += 1
                dispatch_gpu = True

            if is_dispatched < 2:
                if dispatch_cpu:
                    heapq.heappush(mixedg_q, (abs(g_p - 5), -g_num_global_work_items, g_i))
                else:
                    heapq.heappush(mixedc_q, (abs(c_p - 5), -c_num_global_work_items, c_i))


        if is_dispatched == 0:
            if c_num_global_work_items < g_num_global_work_items:
                now_kernel = (c_i, c_p)
                heapq.heappush(mixedg_q, (abs(g_p-5), -g_num_global_work_items, g_i))
            else:
                now_kernel = (g_i, g_p)
                heapq.heappush(mixedc_q, (abs(c_p-5), -c_num_global_work_items, c_i))
    elif fw.nCPU > 0 and fw.nGPU > 0 and mixedc_q:
        i_p, i_num_global_work_items, i = heapq.heappop(mixedc_q)
        i_num_global_work_items = kernels[i].get_num_global_work_items()
        i_p = kernels[i].partition
        now_kernel = (i, i_p)
    elif fw.nCPU > 0 and fw.nGPU > 0 and mixedg_q:
        i_p, i_num_global_work_items, i = heapq.heappop(mixedg_q)
        i_num_global_work_items = kernels[i].get_num_global_work_items()
        i_p = kernels[i].partition
        now_kernel = (i, i_p)
    elif fw.nCPU > 0 and cpu_q:
        c_p, c_num_global_work_items, c_i = heapq.heappop(cpu_q)
        now_kernel = (c_i, 0)
    elif fw.nGPU > 0 and gpu_q:
        g_p, g_num_global_work_items, g_i = heapq.heappop(gpu_q)
        now_kernel = (g_i, 10)
    if now_kernel is not None:
        i, p = now_kernel
        logging.debug( "Selecting kernel " + kernels[i].name + " with partition class value " + str(p))
    return now_kernel, next_kernel


def select_main(kernels, select=baseline_select):
    cmd_qs, ctxs, gpus, cpus = fw.host_initialize(4, 2)
    aCPU, aGPU = fw.nCPU, fw.nGPU
    cpu_q, gpu_q, mixed_q, mixedg_q, mixedc_q = [], [], [], [], []
    num_dispatched = 0
    num_kernels = len(kernels)
    rCPU, rGPU = 0, 0
    num_global_work_items_max = 0
    for i in range(len(kernels)):
        kernels[i].build_kernel(gpus, cpus, ctxs)
        kernels[i].random_data()
        p = kernels[i].partition
        num_global_work_items = kernels[i].get_num_global_work_items()
        if num_global_work_items > num_global_work_items_max:
            num_global_work_items_max = num_global_work_items

        if p == 0:
            heapq.heappush(cpu_q, (p, -num_global_work_items, i))
            rCPU += 1
        elif p == 10:
            heapq.heappush(gpu_q, (-p, -num_global_work_items, i))
            rGPU += 1
        elif p >= 5:
            heapq.heappush(mixedg_q, (abs(p-5), -num_global_work_items, i))
            heapq.heappush(mixed_q, (abs(p-5), -num_global_work_items, i))
            rCPU += 1
            rGPU += 1
        else:
            heapq.heappush(mixedc_q, (abs(p-5), -num_global_work_items, i))
            heapq.heappush(mixed_q, (abs(p-5), -num_global_work_items, i))
            rCPU +=1
            rGPU +=1

    logging.debug( "Task Queue Stats")
    logging.debug(rCPU)
    logging.debug(rGPU)

    logging.debug( "CPU " + str(len(cpu_q)))
    logging.debug( "GPU " + str(len(gpu_q)))
    logging.debug( "Mixed " + str(len(mixed_q)))
    logging.debug( "Mixed CPU " +str(len(mixedc_q)))
    logging.debug( "Mixed GPU " +str(len(mixedg_q)))


    events = []
    now, soon = None, None
    while (len(cpu_q) > 0 or len(gpu_q) > 0 or len(mixed_q) > 0 or num_dispatched != num_kernels) and (len(cpu_q) > 0 or len(gpu_q) > 0 or
                                                                      len(mixedc_q) > 0 or len(mixedg_q) > 0 or num_dispatched != num_kernels):

        logging.debug( "READY DEVICES")
        logging.debug( "GPU " + str(len(fw.ready_queue['gpu'])))
        logging.debug( "CPU " + str(len(fw.ready_queue['cpu'])))
        logging.debug( "Number of tasks left")
        logging.debug( "Mixed Queue " + str(len(mixed_q)))
        logging.debug( "CPU Queue " + str(len(cpu_q)))
        logging.debug( "GPU Queue " + str(len(gpu_q)))
        logging.debug( "Mixed CPU " + str(len(mixedc_q)))
        logging.debug( "Mixed GPU " + str(len(mixedg_q)))

        logging.debug( "Number of available devices (CPU and GPU) " + str(fw.nCPU) + " " + str(fw.nGPU))
        if fw.nCPU > 0 or fw.nGPU > 0:
            logging.debug( "Entering selection phase")
            if soon == None:
                if select is baseline_select or select is look_ahead_select:

                    now, soon = select(kernels, M=mixed_q, G=gpu_q, C=cpu_q)
                else:

                    now, soon = select(kernels, M1=mixedc_q, M2=mixedg_q, G=gpu_q, C=cpu_q, feat_max=num_global_work_items_max, threshold=0.4)
            else:
                now, soon = soon, None

            if now == None:
                logging.debug( "Searching for available devices")
                continue
            i, p = now
            if fw.nCPU > rCPU and fw.nGPU > rGPU:
                logging.debug( "DISPATCH MULTIPLE")

                g_factor = fw.nGPU if rGPU == 0 else fw.nGPU / rGPU
                c_factor = fw.nCPU if rCPU == 0 else fw.nCPU / rCPU
                free_gpus, free_cpus = [], []
                # try:
                while fw.test_and_set(0,1):
                    pass
                if p == 0:
                    for j in range(c_factor):
                        free_cpus.append(fw.ready_queue['cpu'].popleft())
                elif p == 10:
                    for j in range(g_factor):
                        free_gpus.append(fw.ready_queue['gpu'].popleft())
                else:
                    for j in range(c_factor):
                        free_cpus.append(fw.ready_queue['cpu'].popleft())
                    for j in range(g_factor):
                        free_gpus.append(fw.ready_queue['gpu'].popleft())
                fw.rqlock[0] = 0
                # except:
                #     logging.debug( free_cpus, free_gpus, framework.ready_queue, framework.nCPU, framework.nGPU, c_factor, g_factor

                if kernels[i].partition == 0:
                    rCPU -= 1
                elif kernels[i].partition == 10:
                    rGPU -= 1
                else:
                    rCPU -= 1
                    rGPU -= 1

                kernels[i].partition = p
                # kernels[i].build_kernel(gpus, cpus, ctxs)
                # kernels[i].random_data()
                logging.debug( "Dispatching Multiple " + str(kernels[i].name))
                start_time, done_events = kernels[i].dispatch_multiple(free_gpus, free_cpus, ctxs, cmd_qs)
                events.extend(done_events)
                num_dispatched += 1
            # if False:
            #     pass
            else:
                logging.debug( "DISPATCH")
                cpu, gpu = -1, -1
                if p == 0:
                    while fw.test_and_set(0, 1):
                        pass
                    cpu = fw.ready_queue['cpu'].popleft()
                    fw.rqlock[0] = 0
                elif p == 10:
                    while fw.test_and_set(0, 1):
                        pass
                    gpu = fw.ready_queue['gpu'].popleft()
                    fw.rqlock[0] = 0
                else:
                    while fw.test_and_set(0, 1):
                        pass
                    cpu = fw.ready_queue['cpu'].popleft()
                    gpu = fw.ready_queue['gpu'].popleft()
                    fw.rqlock[0] = 0

                if kernels[i].partition == 0:
                    rCPU -= 1
                elif kernels[i].partition == 10:
                    rGPU -= 1
                else:
                    rCPU -=1
                    rGPU -=1

                kernels[i].partition = p

                logging.debug( "Dispatching " + str(kernels[i].name) + " with partition class " + str(kernels[i].partition))
                start_time, done_events = kernels[i].dispatch(gpu, cpu, ctxs, cmd_qs)

                events.extend(done_events)
                num_dispatched +=1
        else:
            logging.debug( "Devices unavailable")
    fw.host_synchronize(cmd_qs, events)

    return fw.dump_device_history()



##########This part is to get optimal partition of the fused kernel
#!/usr/bin/env python

import json, sys, subprocess, os, datetime
import pyschedcl as fw
import argparse
import time
import sys
from decimal import *
# logging.basicConfig(level=logging.DEBUG)
#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


def parse_arg(args=None):
    parser = argparse.ArgumentParser(description='Partition kernels into CPU and GPU heterogeneous system')
    parser.add_argument('-f', '--file',
                        help='Input the json file',
                        required='True')

    parser.add_argument('-d', '--dataset_size',
                        help='Inputs the size of the dataset',
                        default='1024')

    parser.add_argument('-nr', '--runs',
                        help='Number of runs for executing each partitioned variant of the original program',
                        default=5)
    return parser.parse_args(args)


def getoptimalpartition(kernel) :
    #args = parse_arg(sys.argv[1:])


    DEV_NULL = open(os.devnull, 'w')

    p_path = fw.SOURCE_DIR + 'partition/partition.py'
    i_path = fw.SOURCE_DIR + 'info/'
    span_times = []

    for p in range(11):

        execute_cmd = "python " + p_path + " -f " + i_path + kernel.name+ ".json" + " -p " + str(p) + " -d " + args.dataset_size + " > temp.log"

        count = 0
        span = 0
        for r in range(int(args.runs)):
            # print execute_cmd
            sys.stdout.write('\r')

            os.system(execute_cmd)
            sys.stdout.flush()

            time.sleep(1)
            get_span_cmd = "cat temp.log |grep span_time"
            # print get_span_cmd
            span_info = os.popen(get_span_cmd).read().strip("\n")
            if "span_time" in span_info:
                count = count + 1

                profile_time = float(span_info.split(" ")[1])
                if profile_time > 0:
                    span = span + profile_time
        avg_span = span/count
        span_times.append(avg_span)
    bar.finish()
    print("Optimal Partition Class: " + str(span_times.index(min(span_times))) )
    os.system("rm temp.log") 

    return span_times.index(min(span_times))

    # print span_times


from __future__ import print_function
import datetime
import collections
import os
from copy import deepcopy
import framework as fw
import numpy as np
import sys
import json
import re


#https://www.howtoinstall.co/en/ubuntu/trusty/python-pyopencl

# a wrapper function , which takes two kernel instances and does the merging part 
# it will merge them as mentioned by the type of merging and also the id's which are under merging

f = None
orig_stdout =None

def change_stdout(filename):
    #now we redirect stdout to a file
    global f
    global orig_stdout
    orig_stdout = sys.stdout
    f = open(filename, 'w')
    sys.stdout = f


def outer_part(type):
    
    header=""
    if(type==2) :
        header = """
                #define FVALUE .5
                #define BLOCK_SIZE 64
                #define OFFSET 0

                """

    if(type==3) :
        header = header + "#define WORKGROUP_SIZE 1024"

    header =header + """
#define VAL 0
#if defined OP_GRADIENT || defined OP_TOPHAT || defined OP_BLACKHAT
#define EXTRA_PARAMS , __global const uchar * matptr, int mat_step, int mat_offset
#else   
#define EXTRA_PARAMS    
#endif
            """ 
    print(header+'\n\n')



def isalpha(ch):

    if (ch>='a' and ch<='z') or (ch>='A' and ch<='Z') :
        return True
    return False




"""
    This algo is as follows , if kernel2 is dependent on kernel 1 , 
    then output of kernel1 and input of kernel_2 is given the same name  which is combined name of both names 
"""

def T1_fusion(kernel1,kernel2,dep,header,fused_name,body1,body2,buffer1,buffer2) :

    dee =['int','float','void','char','short','long','double']
    body = ""
    stri = """
                int global_x = get_global_id(0);
                    int global_y = get_global_id(1);
                    int local_x =  get_local_id(0);
                    int local_y =  get_local_id(1);
                    int group_x = get_group_id(0);
                    int group_y = get_group_id(1);
                    int ssx, dsx;

                    int block_x=global_x;
                    int y = global_y*2;
                    if ((block_x * 16) >= cols || y >= rows) return;
            
            """

    if dep==-1  :

        #if there is no dependency , just add ,all the buffers to header and append the body parts of both kernels
        body = body + '\t( '+buffer1+ '\n' + buffer2 +')\n'
        
        

        body =body + "\n\t{\n\n"+stri+"\n\t\t"+'{\n\n'

        i =0 
        for x in body1:
            if i==0:
                i = i+1
                continue
            else:
                body = body + '\t\t' + x +'\n'
            

        body =body + '\t\t}\n\n\t\t{\n'

        i=0
        for x in body2 :
            if i==0:
                i = i+1
                continue
            else :
                body = body + '\t\t' +x +'\n'

        body = body +'\t\t}\n\n\t}\n'


    elif dep==1:

        #kernel 2 is dependent on kernel1  , now we have all the buffers 
        inputs_1 = buffer1.split(',')
        inputs_2 = buffer2.split(',')

        dependent_buffers ={}
        #so , take kernels in inputs_2 which are from kernel 1 and replace their names by the buffers in inputs_1 
        for btype in ['input','output','io'] :

            for x in kernel2.buffer_info[btype]:

                #print(x['pos'])
                if 'from' in x: 
                    #print(x['from']['kernel'],end=' ')
                    if x['from']['kernel']==kernel1.id :
                        dependent_buffers[ x['pos'] ] = x['from']['pos']


        #print(dependent_buffers)
        
        buffer_inputs = ""
        for x in inputs_1 :
            
             for y in dee :

                if y in x:
                    buffer_inputs = buffer_inputs + ',\n' + x
                    break



        for i in range(0,len(inputs_2)) :

            if not i in dependent_buffers :
                buffer_inputs = buffer_inputs + ',\n' +inputs_2[i]

        #we remove the first comma
        buffer_inputs =buffer_inputs[1:]


        # now we replace the buffer in kernel2's body with the names that we have ,with the inputs_1 buffers 
        body = body + '\t( ' + buffer_inputs + ')\n'
        body = body + "\n\t{\n\n"+stri+"\n\t\t" +'{\n\n'

        i=0
        for x in body1: 
            if i==0:
                i=i+1
                continue
            else :
                body = body + '\t\t' +x + '\n'


        body =body + '\t\t}\n\n'+"\t\tbarrier(CLK_LOCAL_MEM_FENCE);"+'\n\n\t\t{\n'


        i=0
        for x in body2:

            if i==0:
                i=i+1
                continue

            else :
                
                m=x
                flag=0
                for key in dependent_buffers :
                    
                    l=m 
                    #get the buffer name from these index in inputs_1
                        
                    words = inputs_2[key].split(' ')            
                    buffName = words[len(words)-1]
                    if isalpha(buffName[0]) ==False :
                        buffName = buffName[1:]
                        
                    indices = [m.start() for m in re.finditer(buffName, m)]
                    
                    if len(indices)!=0 :
                        z =l[:indices[0]]
                        j=0
                        flage =0
                        for y in indices :
                    
                            if ( y<0 or y+len(buffName)>len(l) ) or (  ( y>0 and isalpha(l[y-1])==False ) and (  ( y+len(buffName) ) < len(l) and isalpha(l[y+len(buffName)]) ==False )  ):
                                
                                #now this is the word we need to replace
                                replacedword = inputs_1[dependent_buffers[key]].split(' ')
                                replacedBuffer = replacedword[len(replacedword)-1]
                                if isalpha(replacedBuffer[0]) ==False:
                                    replacedBuffer = replacedBuffer[1:]

                                z = z + replacedBuffer 

                                if j<len(indices)-1 :
                                    z = z+ l[y+len(buffName):indices[j+1]]

                                    
                                #replace the string in the line
                                #x = "".join((x[:y],replacedBuffer,x[y+len(buffName):]))
                            else :
                                
                                if(j<len(indices)-1):
                                    z = z+l[y:indices[j+1]]
                                else :
                                    flage=1 

                            j = j+1


                        if flage==1 :
                            z = z+l[ indices[len(indices)-1]:]
                        else :
                            z = z+ l[ indices[len(indices)-1]+len(buffName): ]
                        
                        l= z
                        m=l



                #print(m+'\n\n\n')

                body =body +'\t\t'+ m +'\n'



        body = body +'}\n'





        #check if it is dependent on above kernel 


    else :
        #kernel 1 is dependent on kernel2
        T1_fusion(kernel2,kernel1,1,header,fused_name,body2,body1,buffer2,buffer1)
        return


    
    src = ""
    src = src + outer_part(2)
    print(header)
    print(fused_name)
    print(body)

    src = src + header
    src = src + fused_name
    src = src + body 
    return src



"""
    Here I take the kernels as mentioned in the T1 fusion and just add checking functions 
    which will only execute the kernels if they belong to certain block or else ,the threads in kernel2 
    will wait until the threads in kernel1 are executed , if there is a dependency 
    
"""


def T2_fusion(kernel1,kernel2,dep,header,fused_name,body1,body2,buffer1,buffer2,limits_1,limits_2) :

    body = ""
    stri = """
                int global_x = get_global_id(0);
                    int global_y = get_global_id(1);
                    int local_x =  get_local_id(0);
                    int local_y =  get_local_id(1);
                    int group_x = get_group_id(0);
                    int group_y = get_group_id(1);
                    int ssx, dsx;

                    int block_x=global_x;
                    int y = global_y*2;
                    if ((block_x * 16) >= cols || y >= rows) return;
            
            """

    if dep==-1  :

        #if there is no dependency , just add ,all the buffers to header and append the body parts of both kernels
        body = body + '\t( '+buffer1+ '\n' + buffer2 +')\n'
        
        
        
            #body = body +"\t\t{"
        body =body + "\n\t{\n\n"+stri+'\n\n'
        body= body + "\t\t int X = FVALUE*BLOCK_SIZE*BLOCK_SIZE+OFFSET;\n\n"
        body = body +"\t\t if(local_y*rows + local_x <X)"
        body = body + "\n\t\t"+'{\n\n'

        #print("if()")
        i =0 
        for x in body1:
            if i==0:
                i = i+1
                continue
            else:
                body = body + '\t\t' + x +'\n'
            

        body =body + '\t\t}\n\n'
        body = body + '\t\telse\n\t\t{\n'

        i=0
        for x in body2 :
            if i==0:
                i = i+1
                continue
            else :
                body = body + '\t\t' +x +'\n'

        body = body +'\t\t}\n\n\t}\n'


    elif dep==1:

        #kernel 2 is dependent on kernel1  , now we have all the buffers 
        inputs_1 = buffer1.split(',')
        inputs_2 = buffer2.split(',')

        dependent_buffers ={}
        #so , take kernels in inputs_2 which are from kernel 1 and replace their names by the buffers in inputs_1 
        for btype in ['input','output','io'] :

            for x in kernel2.buffer_info[btype]:

                #print(x['pos'])
                if 'from' in x: 
                    #print(x['from']['kernel'],end=' ')
                    if x['from']['kernel']==kernel1.id :
                        dependent_buffers[ x['pos'] ] = x['from']['pos']


        #print(dependent_buffers)
        
        buffer_inputs = ""
        for x in inputs_1 :
            
            buffer_inputs = buffer_inputs + ',\n' + x

        for i in range(0,len(inputs_2)) :

            if not i in dependent_buffers :
                buffer_inputs = buffer_inputs + ',\n' +inputs_2[i]

        #we remove the first comma
        buffer_inputs =buffer_inputs[1:]


        # now we replace the buffer in kernel2's body with the names that we have ,with the inputs_1 buffers 
        body = body + '\t( ' + buffer_inputs + ')\n'
        body =body + "\n\t{\n\n"+stri+'\n\n'
        body= body + "\t\t int X = FVALUE*BLOCK_SIZE*BLOCK_SIZE+OFFSET;\n\n"
        body = body +"\t\t if(local_y*rows + local_x <X)"
        body = body + "\n\t\t"+'{\n\n'
        i=0
        for x in body1: 
            if i==0:
                i=i+1
                continue
            else :
                body = body + '\t\t' +x + '\n'


        body =body + '\t\t}\n\n'+"\tbarrier(CLK_LOCAL_MEM_FENCE);"+'\n\n'
        body = body +"\t\t if( !(local_y*rows + local_x <X))\n"
        body = body + '\t\t{\n'


        i=0
        for x in body2:

            if i==0:
                i=i+1
                continue

            else :
                
                m=x
                for key in dependent_buffers :
                    
                    l=m 
                    #get the buffer name from these index in inputs_1
                        
                    words = inputs_2[key].split(' ')            
                    buffName = words[len(words)-1]
                    if isalpha(buffName[0]) ==False :
                        buffName = buffName[1:]
                        
                    indices = [m.start() for m in re.finditer(buffName, m)]
                    
                    if len(indices)!=0 :

                        z =l[:indices[0]]
                        j=0
                        for y in indices :
                    
                            if ( y<0 or y+len(buffName)>len(l) ) or (  ( y>0 and isalpha(l[y-1])==False ) and (  ( y+len(buffName) ) < len(l) and isalpha(l[y+len(buffName)]) ==False )  ):
                                
                                #now this is the word we need to replace
                                replacedword = inputs_1[dependent_buffers[key]].split(' ')
                                replacedBuffer = replacedword[len(replacedword)-1]
                                if isalpha(replacedBuffer[0]) ==False:
                                    replacedBuffer = replacedBuffer[1:]

                                z = z + replacedBuffer 

                                if j<len(indices)-1 :
                                    z = z+ l[y+len(buffName):indices[j+1]]

                                    
                                #replace the string in the line
                                #x = "".join((x[:y],replacedBuffer,x[y+len(buffName):]))
                            else :
                                #now here , we just add it without replacing

                                if j<len(indices)-1 :
                                    z = z+ l[y:indices[j+1]]


                            j = j+1

                        z = z+ l[ indices[len(indices)-1]+len(buffName): ]
                        l= z
                        m=l



                #print(m+'\n\n\n')

                body =body +'\t\t'+ m +'\n'



        body = body +'}\n'





        #check if it is dependent on above kernel 


    else :
        #kernel 1 is dependent on kernel2
        T2_fusion(kernel2,kernel1,1,header,fused_name,body2,body1,buffer2,buffer1)
        return


    
    src = ""
    src = src + outer_part(2)
    print(header)
    print(fused_name)
    print(body)

    src = src + header
    src = src + fused_name
    src = src + body 
    return src



"""
    Here I take the kernels as mentioned in the T1 fusion and just add checking functions 
    which will only execute the kernels if they belong to certain block or else ,the threads in kernel2 
    will wait until the threads in kernel1 are executed , if there is a dependency 
    
"""


def T3_fusion(kernel1,kernel2,dep,header,fused_name,body1,body2,buffer1,buffer2,limits_1,limits_2) :

    body = ""
    stri = """
                int global_x = get_global_id(0);
                    int global_y = get_global_id(1);
                    int local_x =  get_local_id(0);
                    int local_y =  get_local_id(1);
                    int group_x = get_group_id(0);
                    int group_y = get_group_id(1);
                    int ssx, dsx;

                    int block_x=global_x;
                    int y = global_y*2;
                    if ((block_x * 16) >= cols || y >= rows) return;
            
            """

    if dep==-1  :

        #if there is no dependency , just add ,all the buffers to header and append the body parts of both kernels
        body = body + '\t( '+buffer1+ '\n' + buffer2 +')\n'
        
        
        
            #body = body +"\t\t{"
        body =body + "\n\t{\n\n"+stri+'\n\n'
        body= body + "\t\t int X = FVALUE*(WORKGROUP_SIZE/BLOCK_SIZE)*(WORKGROUP_SIZE/BLOCK_SIZE)+OFFSET;\n\n"
        body = body +"\t\t if((WORKGROUP_SIZE/BLOCK_SIZE)*group_y + group_x <X)"
        body = body + "\n\t\t"+'{\n\n'

        #print("if()")
        i =0 
        for x in body1:
            if i==0:
                i = i+1
                continue
            else:
                body = body + '\t\t' + x +'\n'
            

        body =body + '\t\t}\n\n'
        body = body + '\t\telse\n\t\t{\n'

        i=0
        for x in body2 :
            if i==0:
                i = i+1
                continue
            else :
                body = body + '\t\t' +x +'\n'

        body = body +'\t\t}\n\n\t}\n'


    elif dep==1:

        #kernel 2 is dependent on kernel1  , now we have all the buffers 
        inputs_1 = buffer1.split(',')
        inputs_2 = buffer2.split(',')

        dependent_buffers ={}
        #so , take kernels in inputs_2 which are from kernel 1 and replace their names by the buffers in inputs_1 
        for btype in ['input','output','io'] :

            for x in kernel2.buffer_info[btype]:

                #print(x['pos'])
                if 'from' in x: 
                    #print(x['from']['kernel'],end=' ')
                    if x['from']['kernel']==kernel1.id :
                        dependent_buffers[ x['pos'] ] = x['from']['pos']


        #print(dependent_buffers)
        
        buffer_inputs = ""
        for x in inputs_1 :
            
            buffer_inputs = buffer_inputs + ',\n' + x

        for i in range(0,len(inputs_2)) :

            if not i in dependent_buffers :
                buffer_inputs = buffer_inputs + ',\n' +inputs_2[i]

        #we remove the first comma
        buffer_inputs =buffer_inputs[1:]


        # now we replace the buffer in kernel2's body with the names that we have ,with the inputs_1 buffers 
        body = body + '\t( ' + buffer_inputs + ')\n'
        body =body + "\n\t{\n\n"+stri+'\n\n'
        body= body + "\t\t\t int X = FVALUE*BLOCK_SIZE*BLOCK_SIZE+OFFSET;\n\n"
        body = body +"\t\t if(((WORKGROUP_SIZE/BLOCK_SIZE)*group_y + group_x <X))"
        body = body + "\n\t\t"+'{\n\n'
        i=0
        for x in body1: 
            if i==0:
                i=i+1
                continue
            else :
                body = body + '\t\t' +x + '\n'


        body =body + '\t\t}\n\n'+"\tbarrier(CLK_LOCAL_MEM_FENCE);"+'\n\n'
        body = body +"\t\t if(!((WORKGROUP_SIZE/BLOCK_SIZE)*group_y + group_x <X))+'\n"
        body = body + '\t\t{\n'


        i=0
        for x in body2:

            if i==0:
                i=i+1
                continue

            else :
                
                m=x
                for key in dependent_buffers :
                    
                    l=m 
                    #get the buffer name from these index in inputs_1
                        
                    words = inputs_2[key].split(' ')            
                    buffName = words[len(words)-1]
                    if isalpha(buffName[0]) ==False :
                        buffName = buffName[1:]
                        
                    indices = [m.start() for m in re.finditer(buffName, m)]
                    
                    if len(indices)!=0 :

                        z =l[:indices[0]]
                        j=0
                        for y in indices :
                    
                            if ( y<0 or y+len(buffName)>len(l) ) or (  ( y>0 and isalpha(l[y-1])==False ) and (  ( y+len(buffName) ) < len(l) and isalpha(l[y+len(buffName)]) ==False )  ):
                                
                                #now this is the word we need to replace
                                replacedword = inputs_1[dependent_buffers[key]].split(' ')
                                replacedBuffer = replacedword[len(replacedword)-1]
                                if isalpha(replacedBuffer[0]) ==False:
                                    replacedBuffer = replacedBuffer[1:]

                                z = z + replacedBuffer 

                                if j<len(indices)-1 :
                                    z = z+ l[y+len(buffName):indices[j+1]]

                                    
                                #replace the string in the line
                                #x = "".join((x[:y],replacedBuffer,x[y+len(buffName):]))
                            else :
                                #now here , we just add it without replacing

                                if j<len(indices)-1 :
                                    z = z+ l[y:indices[j+1]]


                            j = j+1

                        z = z+ l[ indices[len(indices)-1]+len(buffName): ]
                        l= z
                        m=l



                #print(m+'\n\n\n')

                body =body +'\t\t'+ m +'\n'



        body = body +'}\n'





        #check if it is dependent on above kernel 


    else :
        #kernel 1 is dependent on kernel2
        T2_fusion(kernel2,kernel1,1,header,fused_name,body2,body1,buffer2,buffer1)
        return


    src = ""
    src = src + outer_part(2)
    print(header)
    print(fused_name)
    print(body)

    src = src + header
    src = src + fused_name
    src = src + body 
    return src


"""

        Here is the algorithm , i was thinking of . 
        We take two codes and kernels and i merge them by doing this
        I replace the dependent buffer with some name and i will replace its name as it is , in the two source codes and i will
        merge them one by one
    
"""



def kernel_fusion(kernel1, kernel2,type,limits_1,limits_2) :


    #read sources from a kernel1 and kernel2 
    file1 = open(kernel1.src,"r+")
    file2 = open(kernel2.src,"r+")
    kernel_src_1 = file1.readlines()
    kernel_src_2 = file2.readlines()

    header_functions = ""
    x = ""
    z = ""
    source_1 = []
    source_2 = []
    buffer1 = ""
    buffer2 = ""


    # First  get the body of Kernel 1 and Kernel 2
    flag =0
    i=0

    for y in kernel_src_1 :

        if "__kernel" in y and kernel1.name in y :
            break
        header_functions  = header_functions +'\n\n\t'+ y
        i= i+1
        

    
    #now scrap off name from the line 
    wordEndIndex = kernel_src_1[i].index(kernel1.name) + len(kernel1.name) 
    buffer1 = kernel_src_1[i][wordEndIndex+1:]

    j=i+1
    for k in range(j,len(kernel_src_1)) :
        
        if '{' in kernel_src_1[k] :
            i = i+1
            break

        buffer1 = buffer1 + "\t\t"+ kernel_src_1[k]
        i=i+1

    
    for k in range(i,len(kernel_src_1)):    

        source_1.append(kernel_src_1[k])
        x = x+'\n\n\t'+kernel_src_1[k]



    flag =0
    i=0

    for y in kernel_src_2 :

        if "__kernel" in y and kernel2.name in y :
            break
        header_functions  = header_functions +'\n\n\t'+ y
        i=i+1
        

    #now scrap off name from the line 
    wordEndIndex = kernel_src_2[i].index(kernel2.name) + len(kernel2.name) 

    buffer2 = kernel_src_2[i][wordEndIndex+1:]
    #print(buffer2)

    j=i+1
    for k in range(j,len(kernel_src_2)) :
        
        if '{' in kernel_src_2[k] :
            i=i+1
            break

        buffer2 = buffer2 + "\t\t" +kernel_src_2[k]
        
        i=i+1

    
    for k in range(i,len(kernel_src_2)):    

        source_2.append(kernel_src_2[k])
        z = z+'\n\n\t'+kernel_src_2[k]


    buffer1 = buffer1[:-1]
    buffer2 = buffer2[:-1]


    #now we see , if kernel 1 is dependent on kernel 2 or not
    dependency = -1


    if len(kernel1.kernel_deps)>0 and kernel1.kernel_deps[0] == kernel2.id :
        dependency = 2
    elif len(kernel2.kernel_deps)>0 and kernel2.kernel_deps[0] == kernel1.id :
        dependency = 1


    #print(dependency)

    #Routine code fot function name and all
    #body= "{\n\n\t{\n" + x  + "\t}\n\n\n" + "\t{\n" + z + "\t}\n\n}\n"
    fused_name = "__kernel "+"void "+"fused_T1_"+kernel1.name + '_'+ kernel2.name+" \n"
    #print the header data in the file
    #outer_part()
    #print(header_functions)
    #print(fused_name)
    #print(body)

    src = " "
    if type==1 :
        src = T1_fusion(kernel1,kernel2,dependency,header_functions,fused_name,source_1,source_2,buffer1,buffer2)

    elif type==2:
        src = T2_fusion(kernel1,kernel2,dependency,header_functions,fused_name,source_1,source_2,buffer1,buffer2,limits_1,limits_2)
    else :
        src =T3_fusion(kernel1,kernel2,dependency,header_functions,fused_name,source_1,source_2,buffer1,buffer2,limits_1,limits_2)



    #now we will merge their kernels data 
    # we will just fuse the kernels data in kernel1 

    kernel1.id = kernel1.id
    kernel1.name = kernel1.name + "_" + kernel2.name

    kernel1.src = src
    kernel1.work_dimension = max( kernel1.work_dimension , kernel2.work_dimension)

    if len(kernel1.global_work_size ) < len(kernel2.global_work_size) :
        kernel1.global_work_size = kernel2.global_work_size

    if len(kernel1.local_work_size ) < len(kernel2.local_work_size) :
        kernel1.local_work_size = kernel2.local_work_size
    
    if kernel1.i_o == True or kernel2.i_o==True :
        kernel1.i_o = True  

    if kernel1.get_dump_flag()== True or kernel2.get_dump_flag() ==True:
        kernel1.set_dump_flag()

    size = int(0)


    for btype in ['input','io','output'] :
        size = size + len(kernel1.buffer_info[btype])
        kernel1.filenames[btype] = kernel1.filenames[btype] + kernel2.filenames[btype]
        kernel1.buffer_info[btype] =  kernel1.buffer_info[btype]+ kernel2.buffer_info[btype]


        
    return kernel1
    #end of Function




def cal(type,dep) :

    dataset = 64
    partition = 10

    #T1 Fusion 
    filename ="FusedKernelT"+str(type)
    if dep==1 :
        filename= filename+"Dependency.txt"
    else :
        filename =filename+"NoDependency.txt"

    change_stdout(filename)


    if dep== 0:

        info1 = json.loads(open("bilateralFilterKernel.json").read())
        info2 = json.loads(open("depth2vertexKernel.json").read())
        kernel1 = fw.Kernel(info1, dataset=dataset, partition=partition)
        kernel2 = fw.Kernel(info2,dataset=dataset,partition=partition)
        kernel_fused = kernel_fusion(kernel1,kernel2,type,0,0)

    else :
        info3 = json.loads(open("DependentKernel.json").read())
        kernel3 = fw.Kernel(info3[0],dataset=dataset,partition=partition)
        kernel4 = fw.Kernel(info3[1],dataset=dataset,partition=partition)
        kernel_fused = kernel_fusion(kernel3,kernel4,type,0,0)




def main() :

    global orig_stdout
    cal(1,0)
    sys.stdout =orig_stdout
    cal(1,1)
    sys.stdout =orig_stdout
    cal(2,0)
    sys.stdout =orig_stdout
    cal(2,1)
    sys.stdout =orig_stdout
    cal(3,0)
    sys.stdout =orig_stdout
    cal(3,1)
    sys.stdout =orig_stdout



main()


    


    #if type==1 :
        # Type 1 fusion , where 1 kernel executes after other 

        



#####This is the main code

if __name__ == '__main__':
    args = parse_arg(sys.argv[1:])
    task_files = args.file
    kernels = []
    for task in open(task_files,"r").readlines():
        task_src, partition, dataset = task.strip("\n").split(" ")
        info = json.loads(open(fw.SOURCE_DIR + "info/" + task_src).read())
        logging.debug( "Appending kernel" + task_src + " " + partition + " " + dataset)
        kernels.append(fw.Kernel(info, partition=int(partition), dataset=int(dataset)))


    for kernel1 in range(len(kernels)):
        for kernel2 in range(len(kernels)) :

            #fuse these two kernels and

    name = "scheduling_" + args.select +"_" +str(time.time()).replace(".", "")
    dump_dev =None
    if args.dump_output_file:
        fw.dump_output = True
    if args.log:
        f_path = fw.SOURCE_DIR + 'logs/' + name + '_debug.log'
        logging.basicConfig(filename=f_path, level=logging.DEBUG)
    if args.select == "baseline":
        dump_dev = select_main(kernels, select=baseline_select)
    if args.select == "lookahead":
        dump_dev = select_main(kernels, select=look_ahead_select)
    if args.select == "adbias":
        dump_dev = select_main(kernels, select=adaptive_bias_select)

    if args.log:
        f_path = fw.SOURCE_DIR + 'logs/' + name + '_debug.log'
        logging.basicConfig(filename=f_path, level=logging.DEBUG)

    if args.graph:
        filename = fw.SOURCE_DIR + 'gantt_charts/' + name + '.png'
        fw.plot_gantt_chart_graph(dump_dev, filename)
