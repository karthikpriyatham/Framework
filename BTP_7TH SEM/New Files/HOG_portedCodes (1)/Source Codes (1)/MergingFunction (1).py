import datetime
import collections
import os
from copy import deepcopy
import numpy as np

# a wrapper function , which takes two kernel instances and does the merging part 
# it will merge them as mentioned by the type of merging and also the id's which are under merging


#now we redirect stdout to a file 
orig_stdout = sys.stdout
f = open('Output.txt', 'w')
sys.stdout = f


def outer_part():
	
	header ="""
#define VAL 0
#if defined OP_GRADIENT || defined OP_TOPHAT || defined OP_BLACKHAT
#define EXTRA_PARAMS , __global const uchar * matptr, int mat_step, int mat_offset
#else	
#define EXTRA_PARAMS	
#endif
			""" 
	print(header+'\n\n')



def kernel_fusion(kernel1, kernel2,type,limits_1,limits_2) :

	#read sources from a kernel1 and kernel2 
	file1 = open(kernel1['src'],"r+")
	file2 = open(kernel2['src'],"r+")
	kernel_src_1 = file1.readlines()
	kernel_src_2 = file2.readlines()

	header='__kernel void' + kernel_src_1+'_' + kernel_src_2+'('
	
	#now we see , if kernel 1 is dependent on kernel 2 or not
	dependency = -1

	if kernel1['kernel_deps'] == kernel2['id'] :
		dependency = 1
	else if kernel2['kernel_deps']=
	


	fused_name = kernel1['src'][:-3] + '_'+ kernel2['src'][:-3]

	#if type==1 :
		# Type 1 fusion , where 1 kernel executes after other 
		
