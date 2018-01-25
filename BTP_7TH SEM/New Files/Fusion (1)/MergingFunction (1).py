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

		