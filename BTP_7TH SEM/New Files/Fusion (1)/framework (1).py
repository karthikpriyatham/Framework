import pyopencl as cl
import pyopencl.tools
import pyopencl.array
import datetime
import collections
import os
from copy import deepcopy
import mutex
import logging
import numpy as np

numpy_types = {
    "unsigned": np.uint32,
    "unsigned int": np.uint32,
    "uint": np.uint32,
    "int": np.int32,
    "long": np.int64,
    "long int": np.int64,
    "float": np.float32,
    "double": np.float64,
    "char": np.int8,
    "short": np.int16,
    "uchar": np.uint8,
    "unsigned char": np.uint8,
    "ulong": np.uint64,
    "unsigned long": np.uint64,
    "ushort": np.uint16,
    "unsigned short": np.uint16
}

VEC_TYPES = ['char16', 'char2', 'char3', 'char4', 'char8', 'double16', 'double2', 'double3', 'double4', 'double8',
             'float16', 'float2', 'float3', 'float4', 'float8', 'int16', 'int2', 'int3', 'int4', 'int8', 'long16',
             'long2', 'long3', 'long4', 'long8', 'short16', 'short2', 'short3', 'short4', 'short8', 'uchar16', 'uchar2',
             'uchar3', 'uchar4', 'uchar8', 'uint16', 'uint2', 'uint3', 'uint4', 'uint8', 'ulong16', 'ulong2', 'ulong3',
             'ulong4', 'ulong8', 'ushort16', 'ushort2', 'ushort3', 'ushort4', 'ushort8', ]

for datatype in VEC_TYPES:
    numpy_types[datatype] = eval('cl.array.vec.{}'.format(datatype))

nGPU, nCPU = 0, 0
device_history = {"gpu": [], "cpu": []}
ready_queue = {"gpu": collections.deque(), "cpu": collections.deque()}
cs = mutex.mutex()
user_defined = dict()

gpu_offset = 11351.0
gpu_prec = 1e9
cpu_offset = 1487765249.51
cpu_prec = 1e9


def blank_fn(*args, **kwargs):
    pass


class KEvents(object):
    def __init__(self, kernel_name='', kernel_id='', dispatch_id='', dispatch_start=None, dispatch_end=None,
                 write_submit=None, write_start=None, write_end=None, ndrange_start=None, ndrange_end=None,
                 read_start=None, read_end=None):
        self.dispatch_start = dispatch_start
        self.dispatch_end = dispatch_end
        self.write_submit = write_submit
        self.write_start = write_start
        self.write_end = write_end
        self.ndrange_start = ndrange_start
        self.ndrange_end = ndrange_end
        self.read_start = read_start
        self.read_end = read_end
        self.kernel_name = kernel_name
        self.kernel_id = kernel_id
        self.dispatch_id = dispatch_id

    def __str__(self):
        a = deepcopy(self.__dict__)
        for i in a:
            a[i] = str(a[i])
        return str(a)

    def __repr__(self):
        return str(self)

    def normalize(self):
        offset = self.write_submit - self.dispatch_end
        self.write_start -= offset
        self.write_end -= offset
        self.ndrange_start -= offset
        self.ndrange_end -= offset
        self.read_start -= offset
        self.read_end -= offset


def convert_dtime(ts, dev_type):
    if dev_type == 'gpu':
        return datetime.datetime.fromtimestamp((ts * 1.0 / gpu_prec) + gpu_offset)
    elif dev_type == 'cpu':
        return datetime.datetime.fromtimestamp((ts * 1.0 / cpu_prec) + cpu_offset)
    else:
        raise


# TODO: Fix it according to new KEvents.
def log_device_history():
    epoch = datetime.datetime.utcfromtimestamp(0)
    print device_history
    count = 0
    simple_ids = dict()
    for dev in ['gpu', 'cpu']:
        for i in range(len(device_history[dev])):
            for event in device_history[dev][i]:
                if event.kernel_id not in simple_ids:
                    simple_ids[event.kernel_id] = count
                    count += 1
    g_count = len(device_history['gpu'])
    for dev in ['gpu', 'cpu']:
        for i in range(len(device_history[dev])):
            for event in device_history[dev][i]:
                index = g_count if dev == 'cpu' else 0
                print("%d %s_%d %f %f %f" % (
                    index + i, event.kernel_name, simple_ids[event.kernel_id], (event.start - epoch).total_seconds(),
                    (event.end - epoch).total_seconds(), (event.end - event.start).total_seconds()))


# TODO: Handle less than 1000 differently.
def partition_round(elms, percent, exact=-1, total=100, *args, **kwargs):
    """
    Partitions dataset in a predictable way.
    """
    x = elms / 100
    if exact == -1:
        exact = 0 if percent > 50 else 1
    if elms % 2 == 0:
        if percent == 50:
            return elms / 2
        elif exact == 0:
            b = x * (total - percent)
            return partition_round(elms, total) - b if total != 100 else elms - b
        elif exact == 1:
            return x * percent
    else:
        if percent > 50:
            return partition_round(elms - 1, percent, exact, total)
        else:
            return partition_round(elms - 1, percent, exact, total) + 1


part_round = partition_round


def multiple_round(elms, percent, multiples, **kwargs):
    """
    Partition such that the partitioned datasets are multiples of given number.
    """
    for multiple in multiples:
        if elms % multiple == 0 and elms != multiple:
            x = elms / multiple
            return partition_round(x, percent, **kwargs) * multiple


def ctype(dtype):
    """
    Convert a string datatype to corresponding Numpy datatype.
    User can also define new datatypes using user_defined parameter.
    """
    global numpy_types
    try:
        return numpy_types[dtype]
    except:
        Exception("Data Type {} not defined".format(dtype))


def make_ctype(dtype):
    global numpy_types
    if dtype in VEC_TYPES:
        return eval('cl.array.vec.make_{}'.format(dtype))
    else:
        return numpy_types[dtype]


def make_user_defined_dtype(ctxs, name, definition):
    global numpy_types
    if type(definition) is str:
        if name not in numpy_types:
            if definition not in numpy_types:
                raise Exception(
                    "Cant recognize definition {0} should be one of {1}".format(definition, numpy_types.keys()))
            else:
                numpy_types[name] = numpy_types[definition]
        else:
            if numpy_types[definition] != numpy_types[name]:
                raise Exception(
                    "Conflicting definitions {0} and {1} for {2}".format(numpy_types[definition], numpy_types[name],
                                                                         name))
    elif type(definition) is dict:
        raise NotImplementedError
        struct = np.dtype(map(lambda k, v: (k, numpy_types[v]), definition.items()))
        struct, c_decl = cl.tools.match_dtype_to_c_struct(ctxs['gpu'][0])

    else:
        raise Exception('Expected data type definition to be string or dict but got {}'.format(str(type)))


def cs_logic(argument):
    cbid, status, event, kernel, dev_type, dev_no, event_type, callback, args, kwargs = argument
    global cs
    status_map = ['COMPLETE', 'RUNNING', 'SUBMITTED']
    if event_type == 'WRITE' and status_map[status] == 'COMPLETE':
        if len(kernel.clevents[dev_type][dev_no][kwargs['did']]['write']) <= 2:
            fev = kernel.clevents[dev_type][dev_no][kwargs['did']]['write'][0]
            kwargs['ke'].write_start = convert_dtime(fev.get_profiling_info(cl.profiling_info.SUBMIT), dev_type)
            kwargs['ke'].write_end = kwargs['ke'].write_start
            kwargs['ke'].write_submit = kwargs['ke'].write_start
        else:
            fev = kernel.clevents[dev_type][dev_no][kwargs['did']]['write'][0]
            kwargs['ke'].write_submit = convert_dtime(fev.get_profiling_info(cl.profiling_info.SUBMIT), dev_type)
            lev = kernel.clevents[dev_type][dev_no][kwargs['did']]['write'][-2]
            fev = kernel.clevents[dev_type][dev_no][kwargs['did']]['write'][1]
            kwargs['ke'].write_start = convert_dtime(fev.get_profiling_info(cl.profiling_info.START), dev_type)
            kwargs['ke'].write_end = convert_dtime(lev.get_profiling_info(cl.profiling_info.END), dev_type)
            # if dev_no not in kernel.events[dev_type]:
            #     kernel.events[dev_type][dev_no] = []
            # if kernel.events[dev_type][dev_no]:
            #     if kernel.events[dev_type][dev_no][-1].write is None:
            #         kernel.events[dev_type][dev_no][-1].write = convert_dtime(
            #             event.get_profiling_info(cl.profiling_info.SUBMIT), dev_type)
            # else:
            #     ke = KEvents(kernel.name, kernel.id, cbid)
            #     ke.write = convert_dtime(event.get_profiling_info(cl.profiling_info.SUBMIT), dev_type)
            #     kernel.events[dev_type][dev_no].append(ke)
    elif event_type == 'READ' and status_map[status] == 'COMPLETE':
        ready_queue[dev_type].append(dev_no)
        if dev_type == 'gpu':
            global nGPU
            nGPU += 1
            ready_queue['gpu'].append(dev_no)
        else:
            global nCPU
            nCPU += 1
            ready_queue['cpu'].append(dev_no)
        if len(kernel.clevents[dev_type][dev_no][kwargs['did']]['read']) == 1:
            fev = kernel.clevents[dev_type][dev_no][kwargs['did']]['read'][0]
            kwargs['ke'].read_start = convert_dtime(fev.get_profiling_info(cl.profiling_info.SUBMIT), dev_type)
            kwargs['ke'].read_end = kwargs['ke'].read_start
        else:
            fev = kernel.clevents[dev_type][dev_no][kwargs['did']]['read'][0]
            lev = kernel.clevents[dev_type][dev_no][kwargs['did']]['read'][-2]
            kwargs['ke'].read_start = convert_dtime(fev.get_profiling_info(cl.profiling_info.START), dev_type)
            kwargs['ke'].read_end = convert_dtime(lev.get_profiling_info(cl.profiling_info.END), dev_type)
        device_history[dev_type][dev_no].append(kwargs['ke'])
        #
        # if dev_no not in kernel.events[dev_type]:
        #     kernel.events[dev_type][dev_no] = []
        #     ke = KEvents(kernel.name, kernel.id, cbid)
        #     ke.read = convert_dtime(event.get_profiling_info(cl.profiling_info.SUBMIT), dev_type)
        #     kernel.events[dev_type][dev_no].append(ke)
        # kernel.events[dev_type][dev_no][-1].read = convert_dtime(event.get_profiling_info(cl.profiling_info.SUBMIT),
        #                                                          dev_type)
    elif event_type == 'EXEC' and status_map[status] == 'COMPLETE':
        fev = kernel.clevents[dev_type][dev_no][kwargs['did']]['ndrange'][0]
        lev = kernel.clevents[dev_type][dev_no][kwargs['did']]['ndrange'][-2]
        kwargs['ke'].ndrange_start = convert_dtime(fev.get_profiling_info(cl.profiling_info.START), dev_type)
        kwargs['ke'].ndrange_end = convert_dtime(lev.get_profiling_info(cl.profiling_info.END), dev_type)
        # if dev_no not in kernel.events[dev_type]:
        #     kernel.events[dev_type][dev_no] = []
        # if kernel.events[dev_type][dev_no]:
        #     if kernel.events[dev_type][dev_no][-1].ndrange is None:
        #         kernel.events[dev_type][dev_no][-1].ndrange = convert_dtime(
        #             event.get_profiling_info(cl.profiling_info.SUBMIT), dev_type)
        # else:
        #     ke = KEvents(kernel.name, kernel.id, cbid)
        #     ke.ndrange = convert_dtime(event.get_profiling_info(cl.profiling_info.SUBMIT), dev_type)
        #     kernel.events[dev_type][dev_no].append(ke)

        # print nGPU, nCPU
    # kernel.release_buffers(dev_type)
    callback(kernel, dev_type, dev_no, event_type, *args, **kwargs)
    cs.unlock()


def notify_callback(kernel, dev_type, dev_no, event_type, callback=blank_fn, *args, **kwargs):
    """
    Wrapper function that generates and returns a callback function based on parameters.
    """

    logging.info("Set callback %s %s" % (kernel.name, event_type))

    def cb(status):
        cbid = generate_unique_id()
        status_map = ['COMPLETE', 'RUNNING', 'SUBMITTED']
        global cs
        logging.info("Triggered callback %s %s %s %s\n" % (cbid, kernel.name, event_type, status_map[status]))
        cs.lock(cs_logic, (cbid, status, kwargs['event'], kernel, dev_type, dev_no, event_type, callback, args, kwargs))

    return cb


def generate_unique_id():
    """
    Generates and returns a unique id string.
    """
    import uuid
    return str(uuid.uuid1())


class Kernel(object):
    """
    Class to handle all operations perfomed on kernel.
    """

    def __init__(self, src, dataset=1024, partition=None, identifier=None):
        self.dataset = dataset
        if 'id' in src:
            self.id = src['id']
        else:
            self.id = generate_unique_id()

        #### dependent buffer

        if 'depends' in src :
            self.kernel_deps=[]
            for k in src['depends']:
                self.kernel_deps.append(k)
        else :
            self.kernel_deps=[]


        if identifier is not None:
            self.id = identifier
        if 'ecos' in src and str(dataset) in src['ecos']:
            self.eco = src['ecos'][str(dataset)]
        elif 'eco' in src:
            self.eco = src['eco']
        else:
            self.eco = 1
        self.name = src['name']
        self.src = src['src']
        self.partition = src['partition']
        if partition is not None:
            self.partition = partition
        else:
            partition = self.partition
        self.work_dimension = src['workDimension']
        self.global_work_size = src['globalWorkSize']
        if type(self.global_work_size) in [str, unicode]:
            self.global_work_size = eval(self.global_work_size)
        if type(self.global_work_size) is int:
            self.global_work_size = [self.global_work_size]
        if 'localWorkSize' in src:
            self.local_work_size = src['localWorkSize']
        else:
            self.local_work_size = []
        if type(self.local_work_size) in [str, unicode]:
            self.local_work_size = eval(self.local_work_size)
        elif type(self.local_work_size) is int:
            self.local_work_size = [self.local_work_size]
        self.buffer_info = dict()
        if 'inputBuffers' in src:
            self.buffer_info['input'] = src['inputBuffers']
        else:
            self.buffer_info['input'] = []
        if 'outputBuffers' in src:
            self.buffer_info['output'] = src['outputBuffers']
        else:
            self.buffer_info['output'] = []
        if 'ioBuffers' in src:
            self.buffer_info['io'] = src['ioBuffers']
        else:
            self.buffer_info['io'] = []
        self.input_buffers = {'gpu': dict(), 'cpu': dict()}
        self.output_buffers = {'gpu': dict(), 'cpu': dict()}
        self.io_buffers = {'gpu': dict(), 'cpu': dict()}
        self.data = {}
        self.buffer_deps = {}
        if 'varArguments' in src:
            self.variable_args = deepcopy(src['varArguments'])
            self.vargs = src['varArguments']
        else:
            self.variable_args = []
            self.vargs = []
        if 'cpuArguments' in src:
            self.cpu_args = src['cpuArguments']
            print "Ignoring CPU Arguments"
        if 'gpuArguments' in src:
            self.gpu_args = src['gpuArguments']
            print "Ignoring GPU Arguments"
        if 'localArguments' in src:
            self.local_args = src['localArguments']
            for i in range(len(self.local_args)):
                self.local_args[i]['size'] = eval(self.local_args[i]['size'])
        else:
            self.local_args = []
            # self.buffer_info['local'] = deepcopy(self.local_args)
        self.kernel_objects = dict()
        for btype in ['input', 'output', 'io']:
            for i in range(len(self.buffer_info[btype])):

                if type(self.buffer_info[btype][i]['size']) in [str, unicode]:
                    self.buffer_info[btype][i]['size'] = eval(self.buffer_info[btype][i]['size'])
                if 'chunk' in self.buffer_info[btype][i] and type(self.buffer_info[btype][i]['chunk']) in [str,
                                                                                                           unicode]:
                    self.buffer_info[btype][i]['chunk'] = eval(self.buffer_info[btype][i]['chunk'])
                self.buffer_info[btype][i]['create'] = True
                self.buffer_info[btype][i]['enq_write'] = True
                self.buffer_info[btype][i]['enq_read'] = True
                if 'from' in self.buffer_info[btype][i]:
                    self.buffer_deps[self.buffer_info[btype][i]['pos']] = (self.buffer_info[btype][i]['from']['kernel'],
                                                                           self.buffer_info[btype][i]['from']['pos'])

        self.partition_multiples = self.get_partition_multiples()
        self.events = {'gpu': dict(), 'cpu': dict()}
        self.source = None
        self.clevents = {'gpu': dict(), 'cpu': dict()}

    # TODO: Modify to handle dependent buffers.
    def release_buffers(self, obj):
        for i, buff in self.input_buffers[obj].iteritems():
            buff.release()
        for i, buff in self.output_buffers[obj].iteritems():
            buff.release()
        for i, buff in self.io_buffers[obj].iteritems():
            buff.release()

    def eval_vargs(self, partition=None, size_percent=0, offset_percent=0, reverse=False, exact=-1, total=100):
        """
        Method to evaluate kernel arguments.
        """

        def partition_round(elms, percent, exact=exact, total=total):
            return part_round(elms, percent, exact, total)

        if partition is not None:
            size_percent = partition * 10
            offset_percent = 0
            if reverse:
                offset_percent = partition * 10
                partition = 10 - partition
                size_percent = partition * 10
        dataset = self.dataset
        if self.vargs:
            for i in range(len(self.vargs)):
                if type(self.vargs[i]['value']) in [str, unicode]:
                    self.variable_args[i]['value'] = eval(self.vargs[i]['value'])

    def get_partition_multiples(self):
        """
        Determines partition multiples based on work dimension.
        """
        multiples = [1]
        if self.work_dimension == 1:
            if not self.local_work_size:
                multiples = [1]
            else:
                multiples = [self.local_work_size[0], 1]
        elif self.work_dimension == 2:
            if not self.local_work_size:
                multiples = [self.global_work_size[1], 1]
            else:
                multiples = [self.global_work_size[1], self.local_work_size[0], 1]
        elif self.work_dimension == 3:
            if not self.local_work_size:
                multiples = [self.global_work_size[1] * self.global_work_size[2], self.global_work_size[1], 1]
        else:
            print("Invalid Work Dimension")
        return multiples

    def build_kernel(self, gpus, cpus, ctxs):
        """
        Builds Kernels from src/ and stores them in self.kernel_objects dict.
        """
        if not os.path.exists('src/' + self.src):
            raise IOError('Kernel Source File src/%s not Found' % self.src)
        self.source = open('src/' + self.src).read()
        programs = dict()
        for key in ctxs.keys():
            if ctxs[key] is not None:
                programs[key] = cl.Program(ctxs[key], self.source)
        if len(gpus) != 0:
            programs['gpu'].build(devices=gpus)
        if len(cpus) != 0:
            programs['cpu'].build(devices=cpus)

        for key in programs.keys():
            self.kernel_objects[key] = cl.Kernel(programs[key], self.name)
        
        return programs

    def random_data(self, hi=4096):
        """
        Generates random data numpy arrays according to buffer type so that it can be enqueued to buffer.
        Can be used for testing. Will not generate random data for those buffers that are already enqueued.
        Creates empty arrays for read-only buffers.
        """
        import numpy as np
        integers = ['int', 'uint', 'unsigned', 'long', 'unsigned int', 'long int', 'int16', 'int2', 'int3', 'int4',
                    'int8', 'long16', 'long2', 'long3', 'long4', 'long8', 'short16', 'short2', 'short3', 'short4',
                    'short8', 'uint16', 'uint2', 'uint3', 'uint4', 'uint8', 'ulong16', 'ulong2', 'ulong3',
                    'ulong4', 'ulong8', 'ushort16', 'ushort2', 'ushort3', 'ushort4', 'ushort8']
        characters = ['char16', 'char2', 'char3', 'char4', 'char8', 'uchar16', 'uchar2',
                      'uchar3', 'uchar4', 'uchar8']
        for btype in ['input', 'io']:
            self.data[btype] = []
            for i in range(len(self.buffer_info[btype])):

                if not self.buffer_info[btype][i]['enq_write']:
                    self.data[btype].append(None)
                
                elif self.buffer_info[btype][i]['type'] in integers:
                    self.data[btype].append(np.random.randint(hi, size=[self.buffer_info[btype][i]['size']]).astype(
                        ctype(self.buffer_info[btype][i]['type']), order='C'))
                
                elif self.buffer_info[btype][i]['type'] in characters:
                    self.data[btype].append(np.random.randint(128, size=[self.buffer_info[btype][i]['size']]).astype(
                        ctype(self.buffer_info[btype][i]['type']), order='C'))
                
                else:
                    self.data[btype].append(np.random.rand(self.buffer_info[btype][i]['size']).astype(
                        ctype(self.buffer_info[btype][i]['type']), order='C'))
        
        
        self.data['output'] = []
        
        for i in range(len(self.buffer_info['output'])):
            self.data['output'].append(
                np.zeros(self.buffer_info['output'][i]['size'], dtype=ctype(self.buffer_info['output'][i]['type']),
                         order='C'))


    def load_data(self, data):
        """
        Loads data to input buffers
        """
        import numpy as np
        for key in data.keys():
            self.data[key] = []
            for i in range(len(self.buffer_info[key])):
                self.data[key].append(data[key][i])
        self.data['output'] = []
        for i in range(len(self.buffer_info['output'])):
            self.data['output'].append(
                np.zeros(self.buffer_info['output'][i]['size'], dtype=ctype(self.buffer_info['output'][i]['type']),
                         order='C'))

    def get_data(self, pos):
        """
        Returns data at given position. Used to load dependent data.
        """
        for key in self.buffer_info.keys():
            for i in range(len(self.buffer_info[key])):
                if self.buffer_info[key][i]['pos'] == pos:
                    return self.data[key][i]

    def get_buffer_info_location(self, pos):
        """
        Returns buffer_info location at given position. Used to make reusable buffers.
        """
        for key in self.buffer_info.keys():
            for i in range(len(self.buffer_info[key])):
                if self.buffer_info[key][i]['pos'] == pos:
                    return key, i

    def get_buffer_info(self, pos):
        """
        Returns buffer_info at given position. Used to make reusable buffers.
        """
        key, i = self.get_buffer_info_location(pos)
        return self.buffer_info[key][i]

    def get_buffer(self, pos):
        btype, i = self.get_buffer_info_location(pos)
        if btype is 'input':
            return {'gpu': self.input_buffers['gpu'][i], 'cpu': self.input_buffers['cpu'][i]}
        elif btype is 'io':
            return {'gpu': self.io_buffers['gpu'][i], 'cpu': self.io_buffers['cpu'][i]}
        elif btype is 'output':
            return {'gpu': self.output_buffers['gpu'][i], 'cpu': self.output_buffers['cpu'][i]}
        else:
            raise Exception('Expected buffer to be either input, io or output but got ' + str(btype))

    def get_slice_values(self, buffer_info, size_percent, offset_percent, **kwargs):
        """
        Returns Element offset, size based on size_percent, offset_percent.-
        """
        if 'chunk' in buffer_info:
            partition_multiples = [buffer_info['chunk']] + self.partition_multiples
        else:
            partition_multiples = self.partition_multiples
        if buffer_info['break'] != 1:
            eo = 0
            ne = buffer_info['size']
        else:
            if 'exact' not in kwargs:
                eo = multiple_round(buffer_info['size'], offset_percent, partition_multiples, **kwargs)
                ne = multiple_round(buffer_info['size'], size_percent, partition_multiples, **kwargs)
            else:
                eo = multiple_round(buffer_info['size'], offset_percent, partition_multiples, exact=1)
                ne = multiple_round(buffer_info['size'], size_percent, partition_multiples, **kwargs)
        # print offset_percent, size_percent, self.partition, kwargs
        return eo, ne

    def create_buffers(self, ctx, obj, size_percent=100, offset_percent=0, **kwargs):
        """
        Creates buffer objects.
        :param ctx:
        :param obj:
        :param size_percent:
        :param offset_percent:
        :param kwargs:
        :return:
        """
        logging.debug("Creating Input Buffers")
        # print size_percent, offset_percent
        for i in range(len(self.buffer_info['input'])):
            if self.buffer_info['input'][i]['create']:
                eo, ne = self.get_slice_values(self.buffer_info['input'][i], size_percent, offset_percent, **kwargs)
                # print i, eo, ne, obj
                self.input_buffers[obj][i] = cl.Buffer(ctx, cl.mem_flags.READ_WRITE,
                                                       size=self.data['input'][i][eo:eo + ne].nbytes)
                # self.buffer_info['input'][i]['create'] = False
                # self.input_buffers[obj][i] = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                #  size=self.data['input'][i][eo:eo+ne].nbytes, hostbuf=self.data['input'][i][eo:eo+ne])
        logging.debug("Creating Output Buffers")
        for i in range(len(self.buffer_info['output'])):
            if self.buffer_info['output'][i]['create']:
                eo, ne = self.get_slice_values(self.buffer_info['output'][i], size_percent, offset_percent, **kwargs)
                # print i, eo, ne, obj
                self.output_buffers[obj][i] = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY,
                                                        size=self.data['output'][i][eo:eo + ne].nbytes)
                # self.buffer_info['output'][i]['create'] = False
        logging.debug("Creating IO Buffers")
        for i in range(len(self.buffer_info['io'])):
            if self.buffer_info['io'][i]['create']:
                eo, ne = self.get_slice_values(self.buffer_info['io'][i], size_percent, offset_percent, **kwargs)
                self.io_buffers[obj][i] = cl.Buffer(ctx, cl.mem_flags.READ_WRITE,
                                                    size=self.data['io'][i][eo:eo + ne].nbytes)
                # self.buffer_info['io'][i]['create'] = False
                # self.io_buffers[obj].append(cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                #  size=self.buffer_info['io'][i]['size'], hostbuf=self.data['input'][i][eo:eo+ne]))

    # TODO: Test Local Arguments Implementation.
    def set_kernel_args(self, obj):
        """
        Sets Kernel Arguments.
        """
        for i in range(len(self.input_buffers[obj])):
            self.kernel_objects[obj].set_arg(self.buffer_info['input'][i]['pos'], self.input_buffers[obj][i])
        for i in range(len(self.output_buffers[obj])):
            self.kernel_objects[obj].set_arg(self.buffer_info['output'][i]['pos'], self.output_buffers[obj][i])
        for i in range(len(self.io_buffers[obj])):
            self.kernel_objects[obj].set_arg(self.buffer_info['io'][i]['pos'], self.io_buffers[obj][i])
        for i in range(len(self.variable_args)):
            if type(self.variable_args[i]['value']) is list:
                self.kernel_objects[obj].set_arg(self.variable_args[i]['pos'],
                                                 make_ctype(self.variable_args[i]['type'])(
                                                     *self.variable_args[i]['value']))
            else:
                self.kernel_objects[obj].set_arg(self.variable_args[i]['pos'],
                                                 make_ctype(self.variable_args[i]['type'])(
                                                     self.variable_args[i]['value']))
        for i in range(len(self.local_args)):
            self.kernel_objects[obj].set_arg(self.local_args[i]['pos'],
                                             cl.LocalMemory(make_ctype(self.local_args[i]['type'])().nbytes * (
                                                 self.local_args[i]['size'])))

    def enqueue_write_buffers(self, queue, q_id, obj, size_percent=100, offset_percent=0, deps=None, **kwargs):
        """
        Enqueue Write Buffers.

        :param queue:
        :param q_id:
        :param obj:
        :param size_percent:
        :param offset_percent:
        :param deps:
        :param kwargs:
        :return:
        """
        iev, ioev = [None] * len(self.input_buffers[obj]), [None] * len(self.io_buffers[obj])
        depends = [None] * len(self.input_buffers[obj]) + [None] * len(self.io_buffers[obj])
        if len(depends) == 0:
            depends = [None]
        if deps:
            depends[0] = deps
        logging.debug("Enqueuing Write Buffers")
        # print size_percent, offset_percent
        start_barrier_event = cl.enqueue_barrier(queue, wait_for=depends[0])
        for i in range(len(self.input_buffers[obj])):
            if self.buffer_info['input'][i]['enq_write']:
                eo, ne = self.get_slice_values(self.buffer_info['input'][i], size_percent, offset_percent, **kwargs)
                # print i, eo, ne, obj
                iev[i] = cl.enqueue_copy(queue, self.input_buffers[obj][i], self.data['input'][i][eo:eo + ne],
                                         is_blocking=False, wait_for=depends[i])
        if self.input_buffers[obj]:
            depends = [None] * len(self.io_buffers[obj])
        j = len(self.input_buffers[obj])
        for i in range(len(self.io_buffers[obj])):
            if self.buffer_info['io'][i]['enq_write']:
                eo, ne = self.get_slice_values(self.buffer_info['io'][i], size_percent, offset_percent, **kwargs)
                ioev[i] = cl.enqueue_copy(queue, self.io_buffers[obj][i], self.data['io'][i][eo:eo + ne],
                                          is_blocking=False, wait_for=depends[i + j])

        barrier_event = cl.enqueue_barrier(queue)
        barrier_event.set_callback(cl.command_execution_status.COMPLETE,
                                   notify_callback(self, obj, q_id, 'WRITE', event=barrier_event, **kwargs))

        try:
            self.clevents[obj][q_id][kwargs['did']] = dict()
        except KeyError:
            self.clevents[obj][q_id] = dict()
            self.clevents[obj][q_id][kwargs['did']] = dict()
        self.clevents[obj][q_id][kwargs['did']]['write'] = [start_barrier_event] + iev + ioev + [barrier_event]
        return barrier_event




    # TODO: Global Work Offset ?
    def enqueue_nd_range_kernel(self, queue, q_id, obj, size_percent=100, offset_percent=0, **kwargs):
        """
        Enqueue ND Range Kernel

        :param queue:
        :param q_id:
        :param obj:
        :param size_percent:
        :param offset_percent:
        :param kwargs:
        :return:
        """
        global_work_size = []
        global_work_offset = [0] * len(self.global_work_size)
        for i in self.global_work_size:
            global_work_size.append(i)
        global_work_size[0] = multiple_round(global_work_size[0], size_percent, self.partition_multiples, **kwargs)
        # global_work_offset[0] = multiple_round(global_work_size[0], offset_percent,self.partition_multiples, **kwargs)
        if self.local_work_size:
            ev = cl.enqueue_nd_range_kernel(queue, self.kernel_objects[obj], self.global_work_size,
                                            self.local_work_size)
        else:
            ev = cl.enqueue_nd_range_kernel(queue, self.kernel_objects[obj], self.global_work_size, None)
        barrier_event = cl.enqueue_barrier(queue)
        barrier_event.set_callback(cl.command_execution_status.COMPLETE,
                                   notify_callback(self, obj, q_id, 'EXEC', event=barrier_event, **kwargs))

        self.clevents[obj][q_id][kwargs['did']]['ndrange'] = [ev, barrier_event]
        return barrier_event

    # TODO: Remove unnecessary event objects.
    def enqueue_read_buffers(self, queue, q_id, obj, size_percent=100, offset_percent=0, callback=blank_fn, **kwargs):
        """
        Enqueue Read Buffers.

        :param queue:
        :param q_id:
        :param obj:
        :param size_percent:
        :param offset_percent:
        :param callback:
        :param kwargs:
        :return:
        """
        oev, ioev = [None] * len(self.output_buffers[obj]), [None] * len(self.io_buffers[obj])
        logging.debug("Enqueuing Read Buffers")
        # print size_percent, offset_percent
        for i in range(len(self.output_buffers[obj])):
            if self.buffer_info['output'][i]['enq_read']:
                eo, ne = self.get_slice_values(self.buffer_info['output'][i], size_percent, offset_percent, **kwargs)
                # print i, eo, ne, obj
                oev[i] = cl.enqueue_copy(queue, self.data['output'][i][eo:eo + ne], self.output_buffers[obj][i],
                                         is_blocking=False)
        for i in range(len(self.io_buffers[obj])):
            if self.buffer_info['io'][i]['enq_read']:
                eo, ne = self.get_slice_values(self.buffer_info['io'][i], size_percent, offset_percent, **kwargs)
                ioev[i] = cl.enqueue_copy(queue, self.data['io'][i][eo:eo + ne], self.io_buffers[obj][i],
                                          is_blocking=False)
        oev.extend(ioev)
        barrier_event = cl.enqueue_barrier(queue)
        barrier_event.set_callback(cl.command_execution_status.COMPLETE,
                                   notify_callback(self, obj, q_id, 'READ', event=barrier_event, callback=callback,
                                                   **kwargs))
        self.clevents[obj][q_id][kwargs['did']]['read'] = oev + [barrier_event]
        return barrier_event

    def eval_turnaround_time(self, start_time):
        """
        Evaluates and returns turn around time. Time taken for the kernel to finish from start_time

        :param start_time:
        :type start_time: datetime.datetime
        :return: Returns total time in seconds taken for the completion of kernel since start_time
        """
        for event in self.clevents:
            event.wait()
        end_time = start_time
        for key in self.events['gpu'].keys():
            if self.events['gpu'][key][-1].read_end > end_time:
                end_time = self.events['gpu'][key][-1].read_end
        for key in self.events['cpu'].keys():
            if self.events['cpu'][key][-1].read > end_time:
                end_time = self.events['cpu'][key][-1].read_end
        # print end_time, start_time
        return (end_time - start_time).total_seconds()

    # TODO: Do we need to consider wait time between two devices?
    def eval_wait_time(self, start_time):
        """
        Evaluate Wait Time.

        :param start_time:
        :type start_time: datetime.datetime
        :return:
        """
        begin_times = []
        for key in self.events['gpu'].keys():
            begin_times.append(self.events['gpu'][key][0].dispatch_start)
        for key in self.events['cpu'].keys():
            begin_times.append(self.events['cpu'][key][0].dispatch_start)
        begin_time = min(begin_times)
        return (begin_time - start_time).total_seconds()

    def dispatch(self, gpu, cpu, ctxs, cmd_qs, dep=None, partition=None, callback=blank_fn):
        """
        Dispatch Kernel with given partition.
        :param gpu: Denotes the index of gpu device in cmd_qs['gpu'] list or is -1 if we don't want to use device of
         this type.
        :type gpu: Integer
        :param cpu: Denotes the index of cpu device in cmd_qs['cpu'] list or is -1 if we don't want to use device of
         this type.
        :type cpu: Integer
        :param ctxs: Dictionary of Contexts for CPU and GPU devices.
        :type ctxs: dict
        :param cmd_qs: Dictionary of list of Command Queues
        :type cmd_qs: dict
        :param dep:
        :param partition:
        :type partition: Integer from 0 to 10 or None.
        :param callback: A function that will run on the host side once the kernel completes execution on the device.
         Handle unexpected arguments.
        :return:
        """
        dispatch_start = datetime.datetime.now()
        gke = KEvents(self.name, self.id, dispatch_start=dispatch_start)
        cke = KEvents(self.name, self.id, dispatch_start=dispatch_start)

        if partition is not None:
            self.partition = partition
        if dep:
            deps = dep
        else:
            deps = {key: cl.UserEvent(ctxs[key]) for key in ['cpu', 'gpu']}
        if gpu != -1 and cpu != -1:
            size_percent = self.partition * 10
        elif gpu == -1 and cpu != -1:
            size_percent = 0
            self.partition = 0
        elif cpu == -1 and gpu != -1:
            size_percent = 100
            self.partition = 10
        else:
            return None, None
        gdone, cdone = [], []
        if gpu != -1 and self.partition != 0:
            dispatch_id = generate_unique_id()
            gke.dispatch_id = dispatch_id
            global nGPU
            nGPU -= 1
            offset_percent = 0
            self.eval_vargs(size_percent=size_percent, offset_percent=offset_percent)
            self.create_buffers(ctxs['gpu'], 'gpu', size_percent, offset_percent)
            self.set_kernel_args('gpu')
            gdone.append(self.enqueue_write_buffers(cmd_qs['gpu'][gpu], gpu, 'gpu', size_percent, offset_percent,
                                                    deps=[deps['gpu']], ke=gke, did=dispatch_id))
            gdone.append(
                self.enqueue_nd_range_kernel(cmd_qs['gpu'][gpu], gpu, 'gpu', size_percent, 0, ke=gke, did=dispatch_id))
            gdone.append(self.enqueue_read_buffers(cmd_qs['gpu'][gpu], gpu, 'gpu', size_percent, offset_percent,
                                                   callback=callback, ke=gke, did=dispatch_id))
            if gpu not in self.events['gpu']:
                self.events['gpu'][gpu] = []
            self.events['gpu'][gpu].append(gke)
        if cpu != -1 and self.partition != 10:
            dispatch_id = generate_unique_id()
            cke.dispatch_id = dispatch_id
            global nCPU
            nCPU -= 1
            offset_percent = size_percent
            size_percent = 100 - size_percent
            self.eval_vargs(size_percent=size_percent, offset_percent=offset_percent)
            self.create_buffers(ctxs['cpu'], 'cpu', size_percent, offset_percent)
            self.set_kernel_args('cpu')
            cdone.append(self.enqueue_write_buffers(cmd_qs['cpu'][cpu], cpu, 'cpu', size_percent, offset_percent,
                                                    deps=[deps['cpu']], ke=cke, did=dispatch_id))
            cdone.append(
                self.enqueue_nd_range_kernel(cmd_qs['cpu'][cpu], cpu, 'cpu', size_percent, 0, ke=cke, did=dispatch_id))
            cdone.append(self.enqueue_read_buffers(cmd_qs['cpu'][cpu], cpu, 'cpu', size_percent, offset_percent,
                                                   callback=callback, ke=cke, did=dispatch_id))
            if cpu not in self.events['cpu']:
                self.events['cpu'][cpu] = []
            self.events['cpu'][cpu].append(cke)
        if not dep:
            for key in ['gpu', 'cpu']:
                deps[key].set_status(cl.command_execution_status.COMPLETE)
            start_time = datetime.datetime.now()
            gke.dispatch_end = start_time
            cke.dispatch_end = start_time
        return start_time, gdone + cdone

    def dispatch_multiple(self, gpus, cpus, ctxs, cmd_qs, dep=None, partition=None, callback=blank_fn):
        """
        Dispatch Kernel across multiple devices.
        :param gpus:
        :param cpus:
        :param ctxs:
        :param cmd_qs:
        :param dep:
        :param partition:
        :param callback:
        :return:
        """
        dispatch_start = datetime.datetime.now()
        global nGPU
        nGPU -= len(gpus)
        global nCPU
        nCPU -= len(cpus)
        if partition is not None:
            self.partition = partition
        total = len(cpus) + len(gpus)
        size_percent = 100 / total
        if len(gpus) != 0 and len(cpus) != 0:
            gpu_percent = self.partition * 10
        elif len(gpus) == 0 and len(cpus) != 0:
            gpu_percent = 0
            self.partition = 0
        elif len(cpus) == 0 and len(gpus) != 0:
            gpu_percent = 100
            self.partition = 10
        else:
            return None, None
        if gpu_percent == 0:
            nGPU += len(gpus)
        if gpu_percent == 100:
            nCPU += len(cpus)
        cpu_percent = 100 - gpu_percent
        gdone, cdone = [], []
        gkes, ckes = [], []
        if dep:
            deps = dep
        else:
            deps = dict()
            deps['gpu'] = cl.UserEvent(ctxs['gpu'])
            deps['cpu'] = cl.UserEvent(ctxs['cpu'])
        if len(gpus) != 0:
            size_percent = gpu_percent / len(gpus)
        for i in range(len(gpus)):
            dispatch_id = generate_unique_id()
            gke = KEvents(self.name, self.id, dispatch_id, dispatch_start=dispatch_start)
            offset_percent = size_percent * i
            exact = 1
            if i == total - 1:
                size_percent = 100 - offset_percent
                exact = 0
            gpu = gpus[i]
            self.eval_vargs(size_percent=size_percent, offset_percent=offset_percent, exact=exact)
            self.create_buffers(ctxs['gpu'], 'gpu', size_percent, offset_percent, exact=exact)
            self.set_kernel_args('gpu')
            gdone.append(self.enqueue_write_buffers(cmd_qs['gpu'][gpu], gpu, 'gpu', size_percent, offset_percent,
                                                    deps=[deps['gpu']], exact=exact, ke=gke, did=dispatch_id))
            gdone.append(
                self.enqueue_nd_range_kernel(cmd_qs['gpu'][gpu], gpu, 'gpu', size_percent, 0, exact=exact, ke=gke,
                                             did=dispatch_id))
            gdone.append(
                self.enqueue_read_buffers(cmd_qs['gpu'][gpu], gpu, 'gpu', size_percent, offset_percent, exact=exact,
                                          callback=callback, ke=gke, did=dispatch_id))
            if gpu not in self.events['gpu']:
                self.events['gpu'][gpu] = []
            self.events['gpu'][gpu].append(gke)
            gkes.append(gke)

        if len(cpus) != 0:
            size_percent = cpu_percent / len(cpus)
        for i in range(len(cpus)):
            dispatch_id = generate_unique_id()
            cke = KEvents(self.name, self.id, dispatch_id, dispatch_start=dispatch_start)
            exact = 1
            offset_percent = size_percent * i + gpu_percent
            if i == total - 1 - len(gpus):
                size_percent = 100 - offset_percent
                exact = 0
            cpu = cpus[i]
            self.eval_vargs(size_percent=size_percent, offset_percent=offset_percent, exact=exact)
            self.create_buffers(ctxs['cpu'], 'cpu', size_percent, offset_percent, exact=exact)
            self.set_kernel_args('cpu')
            cdone.append(self.enqueue_write_buffers(cmd_qs['cpu'][cpu], cpu, 'cpu', size_percent, offset_percent,
                                                    deps=[deps['cpu']], exact=exact, ke=cke, did=dispatch_id))
            cdone.append(
                self.enqueue_nd_range_kernel(cmd_qs['cpu'][cpu], cpu, 'cpu', size_percent, 0, exact=exact, ke=cke,
                                             did=dispatch_id))
            cdone.append(
                self.enqueue_read_buffers(cmd_qs['cpu'][cpu], cpu, 'cpu', size_percent, offset_percent, exact=exact,
                                          callback=callback, ke=cke, did=dispatch_id))
            if cpu not in self.events['cpu']:
                self.events['cpu'][cpu] = []
            self.events['cpu'][cpu].append(cke)
            ckes.append(cke)
        if not dep:
            for key in ['gpu', 'cpu']:
                deps[key].set_status(cl.command_execution_status.COMPLETE)
            start_time = datetime.datetime.now()
        for ke in gkes + ckes:
            ke.dispatch_end = start_time
        return start_time, gdone + cdone

    def device_requirement(self):
        req = {'gpu': 0, 'cpu': 0, 'all': 0}
        if self.partition > 0:
            req['gpu'] += 1
            req['all'] += 1
        if self.partition < 10:
            req['cpu'] += 1
            req['all'] += 1
        return req


# Deprecated.
def convert_info_to_json(old_file_name, new_file_name=None):
    import json
    if new_file_name is None:
        new_file_name = old_file_name.split('.')[0] + '.json'

    def old_buf(bufs):
        new_bufs = []
        bufs = bufs.split(',')
        temp = [(bufs[i * 4], bufs[i * 4 + 1], bufs[i * 4 + 2], bufs[i * 4 + 3]) for i in range(len(bufs) / 4)]
        for i, j, k, l in temp:
            new_bufs.append({'type': i, 'size': j, 'break': int(k), 'pos': int(l)})
        return new_bufs

    lines = open(old_file_name).readlines()
    clean = map(lambda x: x.strip().split('='), lines)
    clean_dict = dict(clean)
    fresh = dict()
    fresh['name'] = clean_dict['KernelName']
    fresh['src'] = clean_dict['KernelSource']
    fresh['partition'] = int(clean_dict['partition'])
    fresh['workDimension'] = int(clean_dict['workDimension'])
    fresh['globalWorkSize'] = "[" + clean_dict['globalWorkSize'] + "]"
    fresh['inputBuffers'] = old_buf(clean_dict['inputBuffers'])
    fresh['outputBuffers'] = old_buf(clean_dict['outputBuffers'])
    if 'ioBuffers' in clean_dict:
        fresh['ioBuffers'] = old_buf(clean_dict['ioBuffers'])
    if 'varArguments' in clean_dict:
        bufs = clean_dict['varArguments'].split(',')
        fresh['varArguments'] = []
        temp = [(bufs[i * 3], bufs[i * 3 + 1], bufs[i * 3 + 2]) for i in range(len(bufs) / 3)]
        for i, j, k in temp:
            fresh['varArguments'].append({'type': i, 'val': j, 'pos': int(k)})
    if 'vargs' in clean_dict:
        temp = eval(clean_dict['vargs'])
        fresh['vargs'] = []
        for i in range(len(fresh['varArguments'])):
            fresh['vargs'].append(fresh['varArguments'][i])
            fresh['vargs'][-1]['value'] = temp[i]
        del fresh['vargs']
    open(new_file_name, 'w').write(json.dumps(fresh, indent=4))


def get_platform(vendor_name):
    """
    Gets platform given a vendor name
    """
    platforms = cl.get_platforms()
    if len(platforms):
        for pt in cl.get_platforms():
            if vendor_name in pt.name:
                return pt
        print(vendor_name + " Platform not found.")
    else:
        print("No platform found.")


def get_multiple_devices(platform, dev_type, num_devs):
    """
    Get Multiple Devices given a platform and dev type. 
    """
    devs = platform.get_devices(device_type=dev_type)
    if 0 <= num_devs < len(devs):
        print("Requirement: " + str(num_devs) + " greater than availability: " + str(len(devs)))
    else:
        return devs[:num_devs]


def get_single_device(platform, dev_type):
    """
    Get Single Devices given a platform and dev type.
    """
    return get_multiple_devices(platform, dev_type, 1)


def get_sub_devices(platform, dev_type, num_devs, total_compute=16):
    """
    Get Sub Devices given a platform and dev type.
    """
    dev = get_single_device(platform, dev_type)[0]
    return dev.create_sub_devices([cl.device_partition_property.EQUALLY, total_compute / num_devs])


def create_command_queue_for_each(devs, ctx):
    cmd_qs = [cl.CommandQueue(ctx, device=dev, properties=cl.command_queue_properties.PROFILING_ENABLE) for dev in devs]
    return cmd_qs


def host_initialize(num_gpus, num_cpus=2, local=False):
    """
    Set local=True if you device doesn't support GPU. But you still
    want to pretend as if you have one.
    """
    global nGPU
    global nCPU
    if local:
        gpus = None
        cpu_platform = get_platform("Intel(R) OpenCL")
        if num_cpus > 1 and cl.get_cl_header_version() >= (1, 2):
            cpus = get_sub_devices(cpu_platform, cl.device_type.CPU, num_cpus, 4)
        else:
            cpus = get_single_device(cpu_platform, cl.device_type.CPU)
        ctx = cl.Context(devices=cpus)
        ctxs = {"gpu": ctx, "cpu": ctx}
        cmd_q = create_command_queue_for_each(cpus, ctxs['cpu'])
        # cmd_qs = {"gpu":cmd_q, "cpu": cmd_q}
        cmd_qs = {"gpu": [cmd_q[0]], "cpu": [cmd_q[1]]}
        ready_queue['gpu'].append(0)
        ready_queue['cpu'].append(0)
        device_history['gpu'].append([])
        device_history['cpu'].append([])
        gpus = [cpus[0]]
        cpus = [cpus[1]]
        nGPU = 1
        nCPU = 1
        # return cmd_qs, ctxs, [cpus[0]], [cpus[1]]
    else:
        gpus, cpus = [], []
        ctxs = {"gpu": None, "cpu": None}
        cmd_qs = {
            "gpu": [],
            "cpu": []
        }
        if num_gpus > 0:
            gpu_platform = get_platform("NVIDIA CUDA")
            gpus = get_multiple_devices(gpu_platform, cl.device_type.GPU, num_gpus)
            ctxs['gpu'] = cl.Context(devices=gpus)
            cmd_qs['gpu'] = create_command_queue_for_each(gpus, ctxs['gpu'])
        if num_cpus > 0:
            # cpu_platform = get_platform("AMD Accelerated Parallel Processing")
            cpu_platform = get_platform("Intel(R) OpenCL")
            if num_cpus > 1 and cl.get_cl_header_version() >= (1, 2):
                cpus = get_sub_devices(cpu_platform, cl.device_type.CPU, num_cpus)
            else:
                cpus = get_single_device(cpu_platform, cl.device_type.CPU)
            ctxs['cpu'] = cl.Context(devices=cpus)
            cmd_qs['cpu'] = create_command_queue_for_each(cpus, ctxs['cpu'])
        nGPU = len(cmd_qs['gpu'])
        nCPU = len(cmd_qs['cpu'])
        # print nGPU, nCPU
        for key in cmd_qs.keys():
            ready_queue[key].extend(range(len(cmd_qs[key])))
            device_history[key].extend([[] for i in range(len(cmd_qs[key]))])
    return cmd_qs, ctxs, gpus, cpus


def build_kernel_from_info(info_file_name, gpus, cpus, ctxs):
    import json
    info = json.loads(open(info_file_name).read())
    ki = Kernel(info)
    ki.build_kernel(gpus, cpus, ctxs)
    return ki


class KernelDAG(object):
    def __init__(self, srcs, dataset=1024):
        import networkx as nx
        self.kernels = dict()
        self.G = nx.DiGraph()
        self.finished_kernels = set()
        self.free_kernels = list()
        for src in srcs:
            if 'id' not in src:
                raise
            kernel = Kernel(src, dataset, src['partition'])
            self.kernels[kernel.id] = kernel
            self.G.add_node(src['id'], req=kernel.device_requirement())
            if 'depends' in src:
                for i in src['depends']:
                    self.G.add_edge(i, src['id'])
        for node in self.G.nodes():
            if not self.G.predecessors(node):
                self.free_kernels.append(node)
        self.recently_added = self.free_kernels
        # self.loaded = [False] * len(self.kernels)

    def load_dependent_data(self, kernel_id):
        import numpy as np
        deps = self.G.predecessors(kernel_id)
        if len(deps) == 0:
            self.kernels[kernel_id].random_data()
            return
        if len(deps) != 0 and reduce(lambda a, b: a and b, map(lambda s: s in self.finished_kernels, deps)) is False:
            raise
        kernel = self.kernels[kernel_id]
        kernel.random_data()
        for key in ['input', 'io']:
            for i in range(len(kernel.buffer_info[key])):
                if 'from' in kernel.buffer_info[key][i]:
                    data_dep = kernel.buffer_info[key][i]['from']
                    kernel.data[key][i] = self.kernels[data_dep['kernel']].get_data(data_dep['pos'])

    def finished(self, kernel_id, *args, **kwargs):
        self.finished_kernels.add(kernel_id)
        # self.free_kernels.remove(kernel_id)
        successors = self.G.successors(kernel_id)
        # print successors, self.finished_kernels
        self.recently_added = []
        for i in successors:
            # print i, set(self.G.predecessors(i)) ,set(self.G.predecessors(i)) <= set(self.finished_kernels)
            if set(self.G.predecessors(i)) <= set(self.finished_kernels):
                self.recently_added.append(i)
        self.free_kernels.extend(self.recently_added)
        return self.recently_added, self.free_kernels

    def get_free_kernels(self):
        """
        Should return a list of kernels that don't have unmet dependencies.
        """
        return self.free_kernels

    def get_kernel(self, kid):
        """
        Should return a kernel based on kernel id
        """
        return self.kernels.get(kid)


class Task(object):
    import operator

    def __init__(self, kernel):
        """

        :param kernel:
        :type kernel: Kernel
        """
        import operator
        # self.kernels here stands for kernel_ids but not actual kernel objects
        self.kernels = set()
        self.dev_requirement = {'gpu': 0, 'cpu': 0, 'all': 0}
        self.kernels.add(kernel)
        self.finished_kernels = set()
        self.modify_device_requirement(self.kernels, operator.iadd)

    def load_dependent_data_and_buffers(self, dag, kernel_id):
        """

        :param dag:
        :type dag: TaskDAG
        :param kernel_id:
        :return:
        """
        dependencies = dag.get_kernel_parents(kernel_id)
        external_dependencies = set(dependencies) - self.kernels
        kernel = dag.get_kernel(kernel_id)
        for key in ['input', 'io']:
            for i in range(len(kernel.buffer_info[key])):
                if 'from' in kernel.buffer_info[key][i]:
                    data_dep = kernel.buffer_info[key][i]['from']
                    if data_dep['kernel'] in external_dependencies:
                        kernel.data[key][i] = dag.get_kernel(data_dep['kernel']).get_data(data_dep['pos'])
                    elif data_dep['kernel'] in self.get_kernel_ids():
                        dependency = dag.get_kernel(data_dep['kernel'])
                        dbuff = dependency.get_buffer(data_dep['pos'])
                        if key is 'input':
                            kernel.input_buffers['gpu'][i] = dbuff['gpu']
                            kernel.input_buffers['cpu'][i] = dbuff['cpu']
                        elif key is 'io':
                            kernel.io_buffers['gpu'][i] = dbuff['gpu']
                            kernel.io_buffers['cpu'][i] = dbuff['cpu']

    def prepare_kernel(self, kid, dag):
        """
        Prepares Kernel by modifying properties of its buffers, so that buffers can be reused.
        :param kid:
        :param dag:
        :return:
        """
        dependents = dag.get_kernel_children(kid)
        kernel = dag.get_kernel(kid)
        internal_dependents = self.kernels & set(dependents)
        for dep in internal_dependents:
            dependent_kernel = dag.get_kernel(dep)
            for i in dependent_kernel.buffer_deps:
                if dependent_kernel.buffer_deps[i][0] == kid:
                    dep_btype, dep_loc = dependent_kernel.get_buffer_info_location(i)
                    pos = dependent_kernel.buffer_deps[i][1]
                    btype, j = kernel.get_buffer_info_location(pos)
                    if btype == 'output':
                        binfo = kernel.buffer_info['output'].pop(j)
                        kernel.buffer_info['io'].append(binfo)
                    elif btype == 'input' and dep_btype == 'io':
                        binfo = kernel.buffer_info['input'].pop(j)
                        kernel.buffer_info['io'].append(binfo)
                    dependent_kernel.buffer_info[dep_btype][dep_loc]['create'] = False
                    dependent_kernel.buffer_info[dep_btype][dep_loc]['enq_write'] = False

    def prepare_kernels(self, dag):
        """
        Prepares Kernels by modifying properties of their buffers, so that buffers can be reused.
        :param dag:
        :return:
        """
        for kernel in self.get_kernels_sorted(dag):
            self.prepare_kernel(kernel.id, dag)

    def build_kernels(self, gpus, cpus, ctxs):
        """
        Build all kernels in the task.
        :param dag:
        :param gpus:
        :param cpus:
        :param ctxs:
        :return:
        """
        for kernel in self.get_kernels():
            kernel.build_kernel(gpus, cpus, ctxs)

    def dispatch_single(self, dag, gpu, cpu, ctxs, cmd_qs, *args, **kwargs):
        """
        Dispatches all kernels to a given single device. Assumes appropriate data is already loaded.
        """
        if self.is_finished():
            raise Exception("No kernel left in the task to dispatch.")
        if gpu == -1:
            partition = 0
        elif cpu == -1:
            partition = 10
        else:
            raise Exception('Task can only be dispatched to single device.')
        self.prepare_kernels(dag)
        done_events = []
        for kernel in self.get_kernels_sorted(dag):
            kernel.random_data()
            self.load_dependent_data_and_buffers(dag, kernel.id)
            s, d = kernel.dispatch(gpu, cpu, ctxs, cmd_qs, partition=partition)
            done_events.extend(d)
        return done_events

    def remove_kernel(self, kernel):
        """
        Removes given kernel from this task.
        """
        import operator
        if kernel in self.kernels:
            self.kernels.remove(kernel)
            self.modify_device_requirement(set(kernel), operator.isub)
        else:
            raise Exception("Given kernel is not a subset of this task")

    def add_kernels_from_task(self, task):
        """
        Merges a child task into itself.
        """
        import operator
        self.kernels.update(task.get_kernels())
        self.modify_device_requirement(task.get_kernels(), operator.iadd)

    def modify_device_requirement(self, kernels, op=operator.iadd):
        for k in kernels:
            dev_req = k.get_device_requirement()
            for key in self.dev_requirement:
                self.dev_requirement[key] = op(self.dev_requirement[key], dev_req.get(key, 0))

    def get_device_requirement(self):
        return self.dev_requirement

    def get_kernels(self):
        return self.kernels

    def get_kernel_ids(self):
        return map(lambda k: k.id, self.get_kernels())

    def get_kernels_sorted(self, dag):
        import networkx as nx
        dag.get_subgraph(map(lambda k: k.id, self.get_kernels()))
        return map(lambda kid: dag.get_kernel(kid),
                   nx.algorithms.topological_sort(dag.get_subgraph(map(lambda k: k.id, self.get_kernels()))))

    def get_kernel_ids_sorted(self, dag):
        return map(lambda k: k.id, self.get_kernels_sorted(dag))

    def is_supertask(self):
        return len(self.get_kernels()) > 1

    def random_data(self):
        pass

    def is_finished(self):
        return self.kernels == self.finished_kernels

    def update_finished_kernels(self, kernel, dag, *args, **kwargs):
        self.finished_kernels.add(kernel)
        successors = dag.get_kernel_children(kernel.id)
        # print successors, self.finished_kernels
        self.recently_added_kernels = []
        for i in successors:
            # print i, set(self.get_kernel_parents(i)) ,set(self.get_kernel_parents(i)) <= set(self.finished_kernels)
            if set(self.get_kernel_parents(i)) <= set(self.finished_kernels):
                self.recently_added_kernels.append(i)
        self.free_kernels.extend(self.recently_added_kernels)
        return self.recently_added_kernels, self.free_kernels



class TaskDAG(object):
    def __init__(self, srcs, dataset=1024):
        import networkx as nx
        self.kernels = dict()
        self.tasks = dict()
        self.skeleton = nx.DiGraph()
        self.finished_kernels = set()
        self.finished_tasks = set()
        self.free_kernels = list()
        self.free_tasks = list()
        for src in srcs:
            if 'id' not in src:
                raise
            kernel = Kernel(src, dataset, src['partition'])
            self.kernels[kernel.id] = kernel
            self.skeleton.add_node(src['id'], req=kernel.device_requirement())
            if 'depends' in src:
                for i in src['depends']:
                    self.skeleton.add_edge(i, src['id'])
        for node in self.skeleton.nodes():
            if not self.get_kernel_parents(node):
                self.free_kernels.append(node)
        mapping = lambda s: Task(self.get_kernel(s))
        self.G = nx.relabel_nodes(self.skeleton, mapping, copy=True)
        for task in self.G.nodes():
            for kid in task.get_kernel_ids():
                self.tasks[kid] = task
            if not self.get_task_parents(task):
                self.free_tasks.append(task)
        self.recently_added_kernels = self.free_kernels
        self.recently_added_tasks = self.free_tasks
        # self.loaded = [False] * len(self.kernels)

    def update_dependencies(self, task):
        """
        Updates task dependencies. Call this whenever a task is modified. Adds or remove edges to task dag based on
        skeleton kernel dag for the given task.
        :param task:
        :return:
        """
        p, c = self.get_task_parents(task), self.get_task_children(task)
        pt, ct = set(), set()
        for kid in task.get_kernel_ids():
            for pkid in self.get_kernel_parents(kid):
                pt.add(self.tasks[pkid])
            for ckid in self.get_kernel_children(kid):
                ct.add(self.tasks[ckid])
        pt -= task
        ct -= task
        for t in pt - p:
            self.G.add_edge(t, task)
        for t in ct - c:
            self.G.add_edge(task, t)
        for t in p - pt:
            self.remove_edge(t, task)
        for t in c - ct:
            self.remove_edge(task, t)

    def get_skeleton_subgraph(self, kernel_ids):
        return self.skeleton.subgraph(kernel_ids)

    def update_finished_kernels(self, kernel_id, *args, **kwargs):
        self.finished_kernels.add(kernel_id)
        # self.free_kernels.remove(kernel_id)
        successors = self.get_kernel_children(kernel_id)
        # print successors, self.finished_kernels
        self.recently_added_kernels = []
        for i in successors:
            # print i, set(self.get_kernel_parents(i)) ,set(self.get_kernel_parents(i)) <= set(self.finished_kernels)
            if set(self.get_kernel_parents(i)) <= set(self.finished_kernels):
                self.recently_added_kernels.append(i)
        self.free_kernels.extend(self.recently_added_kernels)
        return self.recently_added_kernels, self.free_kernels

    def get_finished_tasks(self):
        return self.finished_tasks

    def update_finished_tasks(self, task):
        self.finished_tasks.add(task)
        children = self.get_task_children(task)
        self.recently_added_tasks = []
        for t in children:
            if set(self.get_task_parents(t)) <= set(self.get_finished_tasks()):
                self.recently_added_tasks.append(t)
        self.free_tasks.extend(self.recently_added_tasks)
        return self.recently_added_tasks, self.free_tasks

    def get_free_kernels(self):
        """
        Should return a list of kernels that don't have unmet dependencies.
        """
        return self.free_kernels

    def get_kernel(self, kid):
        """
        Should return a kernel based on kernel id.
        """
        return self.kernels.get(kid)

    def get_kernel_parents(self, kid):
        """
        Should return a list of kernel ids that are predecessors to given kernel.
        """
        return self.skeleton.predecessors(kid)

    def get_kernel_children(self, kid):
        """
        Should return a list of kernel ids that are successors to given kernel.
        """
        return self.skeleton.successors(kid)

    def get_tasks(self):
        return self.G.nodes()

    def get_tasks_sorted(self):
        import networkx as nx
        return nx.algorithms.topological_sort(self.G)

    def get_all_task_dependencies(self):
        return self.G.edges()

    def get_task_parents(self, task):
        return self.G.predecessors(task)

    def get_task_children(self, task):
        return self.G.successors(task)

    def get_free_tasks(self):
        return self.free_tasks

    def merge_tasks(self, t1, t2):
        """

        :param t1:
        :type t1: Task
        :param t2:
        :type t2: Task
        :return:
        """
        # p, c = self.get_task_parents(t2), self.get_task_children(t2)
        # for task in p:
        #     if task is not t1:
        #         self.G.add_edge(task, t1)
        # for task in c:
        #     if task is not t1:
        #         self.G.add_edge(t1, task)
        dependencies = set().union(*[set(self.get_kernel_parents(kernel)) for kernel in t2.get_kernels()])
        if t1.kernels > dependencies:
            t1.add_kernels_from_task(t2)
        else:
            raise Exception('Some dependent kernels are not part of this task.')
        for kid in t2.get_kernel_ids():
            self.tasks[kid] = t1
        self.update_dependencies(t1)
        self.G.remove_node(t2)

    def split_kernel_from_task(self, kernel, task):
        """
        Remove the given kernel from the given task and create a new task from that kernel, update task
        dependencies accordingly. Returns the newly created task.
        :param kernel:
        :type kernel: Kernel
        :param task:
        :type task: Task
        :return:
        """
        task.remove_kernel(kernel)
        t = Task(kernel)
        self.G.add_node(t)
        self.tasks[kernel.id] = t
        self.update_dependencies(task)
        self.update_dependencies(t)
        return t


# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

ALL_COLORS = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 'rgb(148, 103, 189)',
              'rgb(140, 86, 75)', 'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)', 'rgb(23, 190, 207)',
              'rgb(217,217,217)', 'rgb(240,2,127)', 'rgb(253,205,172)', 'rgb(179,205,227)', 'rgb(166,86,40)',
              'rgb(51,160,44)', 'rgb(247,129,191)', 'rgb(253,191,111)', 'rgb(190,186,218)', 'rgb(231,41,138)',
              'rgb(166,216,84)', 'rgb(153,153,153)', 'rgb(166,118,29)', 'rgb(230,245,201)', 'rgb(255,255,204)',
              'rgb(102,102,102)', 'rgb(77,175,74)', 'rgb(228,26,28)', 'rgb(217,95,2)', 'rgb(255,255,179)',
              'rgb(178,223,138)', 'rgb(190,174,212)', 'rgb(253,180,98)', 'rgb(255,217,47)', 'rgb(31,120,180)',
              'rgb(56,108,176)', 'rgb(229,216,189)', 'rgb(251,154,153)', 'rgb(222,203,228)', 'rgb(203,213,232)',
              'rgb(188,128,189)', 'rgb(55,126,184)', 'rgb(231,138,195)', 'rgb(244,202,228)', 'rgb(191,91,23)',
              'rgb(128,177,211)', 'rgb(27,158,119)', 'rgb(229,196,148)', 'rgb(253,218,236)', 'rgb(102,166,30)',
              'rgb(241,226,204)', 'rgb(255,127,0)', 'rgb(252,141,98)', 'rgb(227,26,28)', 'rgb(254,217,166)',
              'rgb(141,160,203)', 'rgb(204,235,197)', 'rgb(117,112,179)', 'rgb(152,78,163)', 'rgb(202,178,214)',
              'rgb(141,211,199)', 'rgb(106,61,154)', 'rgb(253,192,134)', 'rgb(255,255,51)', 'rgb(179,226,205)',
              'rgb(127,201,127)', 'rgb(251,128,114)', 'rgb(255,242,174)', 'rgb(230,171,2)', 'rgb(102,194,165)',
              'rgb(255,255,153)', 'rgb(179,179,179)', 'rgb(179,222,105)', 'rgb(252,205,229)', 'rgb(204,204,204)',
              'rgb(242,242,242)', 'rgb(166,206,227)', 'rgb(251,180,174)']

AC = ALL_COLORS


def gantt_chart(tasks, title='Gantt Chart', bar_width=0.2, showgrid_x=False, showgrid_y=False, height=600, width=900):
    import plotly.figure_factory as ff
    # devices = {'0': 'GPU 1', '1': 'GPU 2', '2': 'GPU 3', '3': 'GPU 4', '4': 'CPU 1', '5': 'CPU 2'}
    devs = {'gpu': ['GPU 1', 'GPU 2', 'GPU 3', 'GPU 4'], 'cpu': ['CPU 1', 'CPU 2']}
    xtasks = []
    for task in tasks:
        print task.id
        for dtype in task.events:
            for dev in task.events[dtype]:
                for ev in task.events[dtype][dev]:
                    ev.normalize()
                    # xtasks.append(
                    #     dict(Task=devs[dtype][dev], Start=ev.dispatch_start, Finish=ev.dispatch_end,
                    #          Name=task.name + '_dispatch'))
                    # logging.info("{0} {1} {2}s".format(devs[dtype][dev], task.name + '_dispatch',
                    #                                    ev.dispatch_end - ev.dispatch_start))
                    xtasks.append(
                        dict(Task=devs[dtype][dev], Start=ev.write_start, Finish=ev.write_end,
                             Name='{}{}_write'.format(task.name, task.id)))
                    logging.info(
                        "{0} {1} {2}s".format(devs[dtype][dev], '{}{}_write'.format(task.name, task.id),
                                              ev.write_end - ev.write_start))
                    xtasks.append(
                        dict(Task=devs[dtype][dev], Start=ev.ndrange_start, Finish=ev.ndrange_end,
                             Name='{}{}_ndrange'.format(task.name, task.id)))
                    logging.info("{0} {1} {2}s".format(devs[dtype][dev], '{}{}_ndrange'.format(task.name, task.id),
                                                       ev.ndrange_end - ev.ndrange_start))
                    xtasks.append(
                        dict(Task=devs[dtype][dev], Start=ev.read_start, Finish=ev.read_end,
                             Name='{}{}_read'.format(task.name, task.id)))
                    logging.info(
                        "{0} {1} {2}s".format(devs[dtype][dev], '{}{}_read'.format(task.name, task.id),
                                              ev.read_end - ev.read_start))
    fig = ff.create_gantt(xtasks, index_col='Name', show_colorbar=True, group_tasks=True, title=title,
                          bar_width=bar_width, showgrid_x=showgrid_x, showgrid_y=showgrid_y,
                          height=height, width=width, colors=AC)
    return fig
