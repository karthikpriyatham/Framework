
__kernel void normalizeBlockHistograms(__global float1 *blockHistograms,const int noHistogramBins,
										 const int rNoOfHOGBlocksX,const int rNoOfHOGBlocksY,
										 const int blockSizeX, int blockSizeY,
										 const int alignedBlockDimX,const int alignedBlockDimY,const int alignedBlockDimZ,
										 const int width,const int height)
{
	int smemLocalHistogramPos, smemTargetHistogramPos, gmemPosBlock, gmemWritePosBlock;

	float* shLocalHistogram = (float*)allShared;

	float localValue, norm1, norm2; float eps2 = 0.01f;

	//smemLocalHistogramPos = __mul24(threadIdx.y, noHistogramBins) + __mul24(threadIdx.z, blockDim.x) * blockDim.y + threadIdx.x;
	  smemLocalHistogramPos =__mul24(get_local_id(0),noHistogramBins) + __mul24(get_local_id(2),get_local_size(0))*get_local_size(1) + get_local_id(0);

	//gmemPosBlock = __mul24(threadIdx.y, noHistogramBins) + __mul24(threadIdx.z, gridDim.x) * __mul24(blockDim.y, blockDim.x) +
	//	threadIdx.x + __mul24(blockIdx.x, noHistogramBins) * blockDim.y + __mul24(blockIdx.y, gridDim.x) * __mul24(blockDim.y, blockDim.x) * blockDim.z;

	  gmemPosBlock = __mul24(get_local_id(1), noHistogramBins) + __mul24(get_local_id(2),get_num_groups(0)) * __mul24(get_local_size(1), get_local_size(0)) +
		get_local_id(0) + __mul24(get_group_id(0), noHistogramBins) * get_local_size(1) + __mul24(get_group_id(1), get_num_groups(0)) * __mul24(get_local_size(1), get_local_size(0)) * get_local_size(2);
	

	//gmemWritePosBlock = __mul24(threadIdx.z, noHistogramBins) + __mul24(threadIdx.y, gridDim.x) * __mul24(blockDim.y, blockDim.x) +
	//	threadIdx.x + __mul24(blockIdx.x, noHistogramBins) * blockDim.y + __mul24(blockIdx.y, gridDim.x) * __mul24(blockDim.y, blockDim.x) * blockDim.z;

	gmemWritePosBlock = __mul24(get_local_id(2), noHistogramBins) + __mul24(get_local_id(1), get_num_groups(0)) * __mul24(get_local_size(1), get_local_size(0)) +
		get_local_id(0) + __mul24(get_group_id(0), noHistogramBins) * get_local_size(1) + __mul24(get_group_id(1), get_num_groups(0)) * __mul24(get_local_size(1), get_local_size(0)) * get_local_size(2);



	localValue = blockHistograms[gmemPosBlock].x;
	shLocalHistogram[smemLocalHistogramPos] = localValue * localValue;

	//if (blockIdx.x == 10 && blockIdx.y == 8)
	if(get_group_id(0) ==10 && get_group_id(1)==8)
	{
		int asasa;
		asasa = 0;
		asasa++;
	}

	//__syncthreads();
	barrier();


	for(unsigned int s = alignedBlockDimZ >> 1; s>0; s>>=1)
	{
		//if (threadIdx.z < s && (threadIdx.z + s) < blockDim.z)
		//{
			//smemTargetHistogramPos = __mul24(threadIdx.y, noHistogramBins) + __mul24((threadIdx.z + s), blockDim.x) * blockDim.y + threadIdx.x;
		
		if (get_local_id(2)< s && (get_local_id(2) + s) < get_local_size(2))
		{
			smemTargetHistogramPos = __mul24(get_local_id(1), noHistogramBins) + __mul24((get_local_id(2) + s), get_local_size(0)) * get_local_size(1) + get_local_id(0);

			shLocalHistogram[smemLocalHistogramPos] += shLocalHistogram[smemTargetHistogramPos];
		}

		//__syncthreads();
		barrier();

	}

	for (unsigned int s = alignedBlockDimY >> 1; s>0; s>>=1)
	{
		//if (threadIdx.y < s && (threadIdx.y + s) < blockDim.y)
		//{
			//smemTargetHistogramPos = __mul24((threadIdx.y + s), noHistogramBins) + __mul24(threadIdx.z, blockDim.x) * blockDim.y + threadIdx.x;

		if (get_local_id(1)< s && (get_local_id(1) + s) < get_local_size(1))
		{
			smemTargetHistogramPos = __mul24( (get_local_id(1) + s), noHistogramBins) + __mul24((get_local_id(2) ), get_local_size(0)) * get_local_size(1) + get_local_id(0);

			shLocalHistogram[smemLocalHistogramPos] += shLocalHistogram[smemTargetHistogramPos];
		}

		//__syncthreads();
		barrier();

	}

	for(unsigned int s = alignedBlockDimX >> 1; s>0; s>>=1)
	{
		//if (threadIdx.x < s && (threadIdx.x + s) < blockDim.x)
		//{
			//smemTargetHistogramPos = __mul24(threadIdx.y, noHistogramBins) + __mul24(threadIdx.z, blockDim.x) * blockDim.y + (threadIdx.x + s);

		if (get_local_id(0)< s && (get_local_id(0) + s) < get_local_size(0))
		{
			smemTargetHistogramPos = __mul24(get_local_id(1), noHistogramBins) + __mul24( get_local_id(2), get_local_size(0)) * get_local_size(1) + (get_local_id(0) + s);


			shLocalHistogram[smemLocalHistogramPos] += shLocalHistogram[smemTargetHistogramPos];
		}

		//__syncthreads();
		barrier();

	}

	//if (blockIdx.x == 5 && blockIdx.y == 4)
	//{
	//	int asasa;
	//	asasa = 0;
	//	asasa++;
	//}

	norm1 = sqrtf(shLocalHistogram[0]) + __mul24(noHistogramBins, blockSizeX) * blockSizeY;
	localValue /= norm1;

	localValue = fminf(0.2f, localValue); //why 0.2 ??

	//__syncthreads();
		barrier();

	shLocalHistogram[smemLocalHistogramPos] = localValue * localValue;

	//__syncthreads();
		barrier();



	for(unsigned int s = alignedBlockDimZ >> 1; s>0; s>>=1)
	{
		//if (threadIdx.z < s && (threadIdx.z + s) < blockDim.z)
		//{
			//smemTargetHistogramPos = __mul24(threadIdx.y, noHistogramBins) + __mul24((threadIdx.z + s), blockDim.x) * blockDim.y + threadIdx.x;
		
		if (get_local_id(2)< s && (get_local_id(2) + s) < get_local_size(2))
		{
			smemTargetHistogramPos = __mul24(get_local_id(1), noHistogramBins) + __mul24((get_local_id(2) + s), get_local_size(0)) * get_local_size(1) + get_local_id(0);

			shLocalHistogram[smemLocalHistogramPos] += shLocalHistogram[smemTargetHistogramPos];
		}

		//__syncthreads();
		barrier();

	}

	
	

	for (unsigned int s = alignedBlockDimY >> 1; s>0; s>>=1)
	{
		//if (threadIdx.y < s && (threadIdx.y + s) < blockDim.y)
		//{
			//smemTargetHistogramPos = __mul24((threadIdx.y + s), noHistogramBins) + __mul24(threadIdx.z, blockDim.x) * blockDim.y + threadIdx.x;

		if (get_local_id(1)< s && (get_local_id(1) + s) < get_local_size(1))
		{
			smemTargetHistogramPos = __mul24( (get_local_id(1)+s), noHistogramBins) + __mul24((get_local_id(2)), get_local_size(0)) * get_local_size(1) + get_local_id(0);

			shLocalHistogram[smemLocalHistogramPos] += shLocalHistogram[smemTargetHistogramPos];
		}

		//__syncthreads();
		barrier();

	}

	


	for(unsigned int s = alignedBlockDimX >> 1; s>0; s>>=1)
	{
		//if (threadIdx.x < s && (threadIdx.x + s) < blockDim.x)
		//{
			//smemTargetHistogramPos = __mul24(threadIdx.y, noHistogramBins) + __mul24(threadIdx.z, blockDim.x) * blockDim.y + (threadIdx.x + s);

		if (get_local_id(0)< s && (get_local_id(0) + s) < get_local_size(0))
		{
			smemTargetHistogramPos = __mul24(get_local_id(1), noHistogramBins) + __mul24( get_local_id(2), get_local_size(0)) * get_local_size(1) + (get_local_id(0) + s);


			shLocalHistogram[smemLocalHistogramPos] += shLocalHistogram[smemTargetHistogramPos];
		}

		//__syncthreads();
		barrier();

	}





	
	for(unsigned int s = alignedBlockDimX >> 1; s>0; s>>=1)
	{
		if (threadIdx.x < s && (threadIdx.x + s) < blockDim.x)
		{
			smemTargetHistogramPos = __mul24(threadIdx.y, noHistogramBins) + __mul24(threadIdx.z, blockDim.x) * blockDim.y + (threadIdx.x + s);
			shLocalHistogram[smemLocalHistogramPos] += shLocalHistogram[smemTargetHistogramPos];
		}

		__syncthreads();
	}

	norm2 = sqrtf(shLocalHistogram[0]) + eps2;
	localValue /= norm2;

	blockHistograms[gmemWritePosBlock].x = localValue;

	//if (blockIdx.x == 10 && blockIdx.y == 8)
	if(get_group_id(0)==10 && get_group_id(1)==8)
	{
		int asasa;
		asasa = 0;
		asasa++;
	}
}
