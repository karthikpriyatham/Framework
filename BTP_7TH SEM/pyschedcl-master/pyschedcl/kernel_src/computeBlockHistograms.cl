int mul24(int a,int b) 
{
 	return a*b; 
}

__kernel void computeBlockHistogramsWithGauss(float2* inputImage, float1* blockHistograms, int noHistogramBins,
												int cellSizeX, int cellSizeY, int blockSizeX, int blockSizeY,
												int leftoverX, int leftoverY, int width, int height)
{
	int i;
	float2 localValue;
	float* shLocalHistograms = (float*)allShared;

	int cellIdx = threadIdx.y;
	int cellIdy = threadIdx.z;
	int columnId = threadIdx.x;

	int smemReadPos = mul24(cellIdx, noHistogramBins) + mul24(cellIdy, blockSizeX) * noHistogramBins;
	int gmemWritePos = mul24(threadIdx.y, noHistogramBins) + mul24(threadIdx.z, gridDim.x) * mul24(blockDim.y, noHistogramBins) +
		mul24(blockIdx.x, noHistogramBins) * blockDim.y + mul24(blockIdx.y, gridDim.x) * mul24(blockDim.y, noHistogramBins) * blockDim.z;

	int gmemReadStride = width;

	int gmemReadPos = leftoverX + mul24(leftoverY, gmemReadStride) +
		(mul24(blockIdx.x, cellSizeX) + mul24(blockIdx.y, cellSizeY) * gmemReadStride)
		+ (columnId + mul24(cellIdx, cellSizeX) + mul24(cellIdy, cellSizeY) * gmemReadStride);

	int histogramSize = mul24(noHistogramBins, blockSizeX) * blockSizeY;
	int smemLocalHistogramPos = (columnId + mul24(cellIdx, cellSizeX)) * histogramSize + mul24(cellIdy, histogramSize) * mul24(blockSizeX, cellSizeX);

	int cmemReadPos = columnId + mul24(cellIdx, cellSizeX) + mul24(cellIdy, cellSizeY) * mul24(cellSizeX, blockSizeX);

	float atx, aty;
	float pIx, pIy, pIz;

	int fIx, fIy, fIz;
	int cIx, cIy, cIz;
	float dx, dy, dz;
	float cx, cy, cz;

	bool lowervalidx, lowervalidy;
	bool uppervalidx, uppervalidy;
	bool canWrite;

	int offset;

	for (i=0; i<histogramSize; i++) shLocalHistograms[smemLocalHistogramPos + i] = 0;

#ifdef UNROLL_LOOPS
	int halfSizeYm1 = cellSizeY / 2 - 1;
#endif

	//if (blockIdx.x == 5 && blockIdx.y == 4)
	//{
	//	int asasa;
	//	asasa = 0;
	//	asasa++;
	//}

	for (i=0; i<cellSizeY; i++)
	{
		localValue = inputImage[gmemReadPos + i * gmemReadStride];
		localValue.x *= tex1D(texGauss, cmemReadPos + i * cellSizeX * blockSizeX);

		atx = cellIdx * cellSizeX + columnId + 0.5;
		aty = cellIdy * cellSizeY + i + 0.5;

		pIx = halfBin[0] - oneHalf + (atx - cenBound[0]) * bandWidth[0];
		pIy = halfBin[1] - oneHalf + (aty - cenBound[1]) * bandWidth[1];
		pIz = halfBin[2] - oneHalf + (localValue.y - cenBound[2]) * bandWidth[2];

		fIx = floorf(pIx); fIy = floorf(pIy); fIz = floorf(pIz);
		cIx = fIx + 1; cIy = fIy + 1; cIz = fIz + 1; //eq ceilf(pI.)

		dx = pIx - fIx; dy = pIy - fIy; dz = pIz - fIz;
		cx = 1 - dx; cy = 1 - dy; cz = 1 - dz;

		cIz %= tvbin[2];
		fIz %= tvbin[2];
		if (fIz < 0) fIz += tvbin[2];
		if (cIz < 0) cIz += tvbin[2];

#ifdef UNROLL_LOOPS
		if ((i & halfSizeYm1) == 0)
#endif
		{
			uppervalidx = !(cIx >= tvbin[0] - oneHalf || cIx < -oneHalf);
			uppervalidy = !(cIy >= tvbin[1] - oneHalf || cIy < -oneHalf);
			lowervalidx = !(fIx < -oneHalf || fIx >= tvbin[0] - oneHalf);
			lowervalidy = !(fIy < -oneHalf || fIy >= tvbin[1] - oneHalf);
		}

		canWrite = (lowervalidx) && (lowervalidy);
		if (canWrite)
		{
			offset = smemLocalHistogramPos + (fIx + fIy * blockSizeY) * noHistogramBins;
			shLocalHistograms[offset + fIz] += localValue.x * cx * cy * cz;
			shLocalHistograms[offset + cIz] += localValue.x * cx * cy * dz;
		}

		canWrite = (lowervalidx) && (uppervalidy);
		if (canWrite)
		{
			offset = smemLocalHistogramPos + (fIx + cIy * blockSizeY) * noHistogramBins;
			shLocalHistograms[offset + fIz] += localValue.x * cx * dy * cz;
			shLocalHistograms[offset + cIz] += localValue.x * cx * dy * dz;
		}

		canWrite = (uppervalidx) && (lowervalidy);
		if (canWrite)
		{
			offset = smemLocalHistogramPos + (cIx + fIy * blockSizeY) * noHistogramBins;
			shLocalHistograms[offset + fIz] += localValue.x * dx * cy * cz;
			shLocalHistograms[offset + cIz] += localValue.x * dx * cy * dz;
		}

		canWrite = (uppervalidx) && (uppervalidy);
		if (canWrite)
		{
			offset = smemLocalHistogramPos + (cIx + cIy * blockSizeY) * noHistogramBins;
			shLocalHistograms[offset + fIz] += localValue.x * dx * dy * cz;
			shLocalHistograms[offset + cIz] += localValue.x * dx * dy * dz;
		}
	}

	__syncthreads();

	//TODO -> aligned block size * cell size
	int smemTargetHistogramPos;
	for(unsigned int s = blockSizeY >> 1; s>0; s>>=1)
	{
		if (cellIdy < s && (cellIdy + s) < blockSizeY)
		{
			smemTargetHistogramPos = (columnId + mul24(cellIdx, cellSizeX)) * histogramSize + mul24((cellIdy + s), histogramSize) * mul24(blockSizeX, cellSizeX);

#ifdef UNROLL_LOOPS
			shLocalHistograms[smemLocalHistogramPos + 0] += shLocalHistograms[smemTargetHistogramPos + 0];
			shLocalHistograms[smemLocalHistogramPos + 1] += shLocalHistograms[smemTargetHistogramPos + 1];
			shLocalHistograms[smemLocalHistogramPos + 2] += shLocalHistograms[smemTargetHistogramPos + 2];
			shLocalHistograms[smemLocalHistogramPos + 3] += shLocalHistograms[smemTargetHistogramPos + 3];
			shLocalHistograms[smemLocalHistogramPos + 4] += shLocalHistograms[smemTargetHistogramPos + 4];
			shLocalHistograms[smemLocalHistogramPos + 5] += shLocalHistograms[smemTargetHistogramPos + 5];
			shLocalHistograms[smemLocalHistogramPos + 6] += shLocalHistograms[smemTargetHistogramPos + 6];
			shLocalHistograms[smemLocalHistogramPos + 7] += shLocalHistograms[smemTargetHistogramPos + 7];
			shLocalHistograms[smemLocalHistogramPos + 8] += shLocalHistograms[smemTargetHistogramPos + 8];
			shLocalHistograms[smemLocalHistogramPos + 9] += shLocalHistograms[smemTargetHistogramPos + 9];
			shLocalHistograms[smemLocalHistogramPos + 10] += shLocalHistograms[smemTargetHistogramPos + 10];
			shLocalHistograms[smemLocalHistogramPos + 11] += shLocalHistograms[smemTargetHistogramPos + 11];
			shLocalHistograms[smemLocalHistogramPos + 12] += shLocalHistograms[smemTargetHistogramPos + 12];
			shLocalHistograms[smemLocalHistogramPos + 13] += shLocalHistograms[smemTargetHistogramPos + 13];
			shLocalHistograms[smemLocalHistogramPos + 14] += shLocalHistograms[smemTargetHistogramPos + 14];
			shLocalHistograms[smemLocalHistogramPos + 15] += shLocalHistograms[smemTargetHistogramPos + 15];
			shLocalHistograms[smemLocalHistogramPos + 16] += shLocalHistograms[smemTargetHistogramPos + 16];
			shLocalHistograms[smemLocalHistogramPos + 17] += shLocalHistograms[smemTargetHistogramPos + 17];
			shLocalHistograms[smemLocalHistogramPos + 18] += shLocalHistograms[smemTargetHistogramPos + 18];
			shLocalHistograms[smemLocalHistogramPos + 19] += shLocalHistograms[smemTargetHistogramPos + 19];
			shLocalHistograms[smemLocalHistogramPos + 20] += shLocalHistograms[smemTargetHistogramPos + 20];
			shLocalHistograms[smemLocalHistogramPos + 21] += shLocalHistograms[smemTargetHistogramPos + 21];
			shLocalHistograms[smemLocalHistogramPos + 22] += shLocalHistograms[smemTargetHistogramPos + 22];
			shLocalHistograms[smemLocalHistogramPos + 23] += shLocalHistograms[smemTargetHistogramPos + 23];
			shLocalHistograms[smemLocalHistogramPos + 24] += shLocalHistograms[smemTargetHistogramPos + 24];
			shLocalHistograms[smemLocalHistogramPos + 25] += shLocalHistograms[smemTargetHistogramPos + 25];
			shLocalHistograms[smemLocalHistogramPos + 26] += shLocalHistograms[smemTargetHistogramPos + 26];
			shLocalHistograms[smemLocalHistogramPos + 27] += shLocalHistograms[smemTargetHistogramPos + 27];
			shLocalHistograms[smemLocalHistogramPos + 28] += shLocalHistograms[smemTargetHistogramPos + 28];
			shLocalHistograms[smemLocalHistogramPos + 29] += shLocalHistograms[smemTargetHistogramPos + 29];
			shLocalHistograms[smemLocalHistogramPos + 30] += shLocalHistograms[smemTargetHistogramPos + 30];
			shLocalHistograms[smemLocalHistogramPos + 31] += shLocalHistograms[smemTargetHistogramPos + 31];
			shLocalHistograms[smemLocalHistogramPos + 32] += shLocalHistograms[smemTargetHistogramPos + 32];
			shLocalHistograms[smemLocalHistogramPos + 33] += shLocalHistograms[smemTargetHistogramPos + 33];
			shLocalHistograms[smemLocalHistogramPos + 34] += shLocalHistograms[smemTargetHistogramPos + 34];
			shLocalHistograms[smemLocalHistogramPos + 35] += shLocalHistograms[smemTargetHistogramPos + 35];
#else
			for (i=0; i<histogramSize; i++)
				shLocalHistograms[smemLocalHistogramPos + i] += shLocalHistograms[smemTargetHistogramPos + i];
#endif
		}

		__syncthreads();
	}

	for(unsigned int s = blockSizeX >> 1; s>0; s>>=1)
	{
		if (cellIdx < s && (cellIdx + s) < blockSizeX)
		{
			smemTargetHistogramPos = (columnId + mul24((cellIdx + s), cellSizeX)) * histogramSize + mul24(cellIdy, histogramSize) * mul24(blockSizeX, cellSizeX);

#ifdef UNROLL_LOOPS
			shLocalHistograms[smemLocalHistogramPos + 0] += shLocalHistograms[smemTargetHistogramPos + 0];
			shLocalHistograms[smemLocalHistogramPos + 1] += shLocalHistograms[smemTargetHistogramPos + 1];
			shLocalHistograms[smemLocalHistogramPos + 2] += shLocalHistograms[smemTargetHistogramPos + 2];
			shLocalHistograms[smemLocalHistogramPos + 3] += shLocalHistograms[smemTargetHistogramPos + 3];
			shLocalHistograms[smemLocalHistogramPos + 4] += shLocalHistograms[smemTargetHistogramPos + 4];
			shLocalHistograms[smemLocalHistogramPos + 5] += shLocalHistograms[smemTargetHistogramPos + 5];
			shLocalHistograms[smemLocalHistogramPos + 6] += shLocalHistograms[smemTargetHistogramPos + 6];
			shLocalHistograms[smemLocalHistogramPos + 7] += shLocalHistograms[smemTargetHistogramPos + 7];
			shLocalHistograms[smemLocalHistogramPos + 8] += shLocalHistograms[smemTargetHistogramPos + 8];
			shLocalHistograms[smemLocalHistogramPos + 9] += shLocalHistograms[smemTargetHistogramPos + 9];
			shLocalHistograms[smemLocalHistogramPos + 10] += shLocalHistograms[smemTargetHistogramPos + 10];
			shLocalHistograms[smemLocalHistogramPos + 11] += shLocalHistograms[smemTargetHistogramPos + 11];
			shLocalHistograms[smemLocalHistogramPos + 12] += shLocalHistograms[smemTargetHistogramPos + 12];
			shLocalHistograms[smemLocalHistogramPos + 13] += shLocalHistograms[smemTargetHistogramPos + 13];
			shLocalHistograms[smemLocalHistogramPos + 14] += shLocalHistograms[smemTargetHistogramPos + 14];
			shLocalHistograms[smemLocalHistogramPos + 15] += shLocalHistograms[smemTargetHistogramPos + 15];
			shLocalHistograms[smemLocalHistogramPos + 16] += shLocalHistograms[smemTargetHistogramPos + 16];
			shLocalHistograms[smemLocalHistogramPos + 17] += shLocalHistograms[smemTargetHistogramPos + 17];
			shLocalHistograms[smemLocalHistogramPos + 18] += shLocalHistograms[smemTargetHistogramPos + 18];
			shLocalHistograms[smemLocalHistogramPos + 19] += shLocalHistograms[smemTargetHistogramPos + 19];
			shLocalHistograms[smemLocalHistogramPos + 20] += shLocalHistograms[smemTargetHistogramPos + 20];
			shLocalHistograms[smemLocalHistogramPos + 21] += shLocalHistograms[smemTargetHistogramPos + 21];
			shLocalHistograms[smemLocalHistogramPos + 22] += shLocalHistograms[smemTargetHistogramPos + 22];
			shLocalHistograms[smemLocalHistogramPos + 23] += shLocalHistograms[smemTargetHistogramPos + 23];
			shLocalHistograms[smemLocalHistogramPos + 24] += shLocalHistograms[smemTargetHistogramPos + 24];
			shLocalHistograms[smemLocalHistogramPos + 25] += shLocalHistograms[smemTargetHistogramPos + 25];
			shLocalHistograms[smemLocalHistogramPos + 26] += shLocalHistograms[smemTargetHistogramPos + 26];
			shLocalHistograms[smemLocalHistogramPos + 27] += shLocalHistograms[smemTargetHistogramPos + 27];
			shLocalHistograms[smemLocalHistogramPos + 28] += shLocalHistograms[smemTargetHistogramPos + 28];
			shLocalHistograms[smemLocalHistogramPos + 29] += shLocalHistograms[smemTargetHistogramPos + 29];
			shLocalHistograms[smemLocalHistogramPos + 30] += shLocalHistograms[smemTargetHistogramPos + 30];
			shLocalHistograms[smemLocalHistogramPos + 31] += shLocalHistograms[smemTargetHistogramPos + 31];
			shLocalHistograms[smemLocalHistogramPos + 32] += shLocalHistograms[smemTargetHistogramPos + 32];
			shLocalHistograms[smemLocalHistogramPos + 33] += shLocalHistograms[smemTargetHistogramPos + 33];
			shLocalHistograms[smemLocalHistogramPos + 34] += shLocalHistograms[smemTargetHistogramPos + 34];
			shLocalHistograms[smemLocalHistogramPos + 35] += shLocalHistograms[smemTargetHistogramPos + 35];
#else
			for (i=0; i<histogramSize; i++)
				shLocalHistograms[smemLocalHistogramPos + i] += shLocalHistograms[smemTargetHistogramPos + i];
#endif
		}

		__syncthreads();
	}

	for(unsigned int s = cellSizeX >> 1; s>0; s>>=1)
	{
		if (columnId < s && (columnId + s) < cellSizeX)
		{
			smemTargetHistogramPos = (columnId + s + mul24(cellIdx, cellSizeX)) * histogramSize + mul24(cellIdy, histogramSize) * mul24(blockSizeX, cellSizeX);

#ifdef UNROLL_LOOPS
			shLocalHistograms[smemLocalHistogramPos + 0] += shLocalHistograms[smemTargetHistogramPos + 0];
			shLocalHistograms[smemLocalHistogramPos + 1] += shLocalHistograms[smemTargetHistogramPos + 1];
			shLocalHistograms[smemLocalHistogramPos + 2] += shLocalHistograms[smemTargetHistogramPos + 2];
			shLocalHistograms[smemLocalHistogramPos + 3] += shLocalHistograms[smemTargetHistogramPos + 3];
			shLocalHistograms[smemLocalHistogramPos + 4] += shLocalHistograms[smemTargetHistogramPos + 4];
			shLocalHistograms[smemLocalHistogramPos + 5] += shLocalHistograms[smemTargetHistogramPos + 5];
			shLocalHistograms[smemLocalHistogramPos + 6] += shLocalHistograms[smemTargetHistogramPos + 6];
			shLocalHistograms[smemLocalHistogramPos + 7] += shLocalHistograms[smemTargetHistogramPos + 7];
			shLocalHistograms[smemLocalHistogramPos + 8] += shLocalHistograms[smemTargetHistogramPos + 8];
			shLocalHistograms[smemLocalHistogramPos + 9] += shLocalHistograms[smemTargetHistogramPos + 9];
			shLocalHistograms[smemLocalHistogramPos + 10] += shLocalHistograms[smemTargetHistogramPos + 10];
			shLocalHistograms[smemLocalHistogramPos + 11] += shLocalHistograms[smemTargetHistogramPos + 11];
			shLocalHistograms[smemLocalHistogramPos + 12] += shLocalHistograms[smemTargetHistogramPos + 12];
			shLocalHistograms[smemLocalHistogramPos + 13] += shLocalHistograms[smemTargetHistogramPos + 13];
			shLocalHistograms[smemLocalHistogramPos + 14] += shLocalHistograms[smemTargetHistogramPos + 14];
			shLocalHistograms[smemLocalHistogramPos + 15] += shLocalHistograms[smemTargetHistogramPos + 15];
			shLocalHistograms[smemLocalHistogramPos + 16] += shLocalHistograms[smemTargetHistogramPos + 16];
			shLocalHistograms[smemLocalHistogramPos + 17] += shLocalHistograms[smemTargetHistogramPos + 17];
			shLocalHistograms[smemLocalHistogramPos + 18] += shLocalHistograms[smemTargetHistogramPos + 18];
			shLocalHistograms[smemLocalHistogramPos + 19] += shLocalHistograms[smemTargetHistogramPos + 19];
			shLocalHistograms[smemLocalHistogramPos + 20] += shLocalHistograms[smemTargetHistogramPos + 20];
			shLocalHistograms[smemLocalHistogramPos + 21] += shLocalHistograms[smemTargetHistogramPos + 21];
			shLocalHistograms[smemLocalHistogramPos + 22] += shLocalHistograms[smemTargetHistogramPos + 22];
			shLocalHistograms[smemLocalHistogramPos + 23] += shLocalHistograms[smemTargetHistogramPos + 23];
			shLocalHistograms[smemLocalHistogramPos + 24] += shLocalHistograms[smemTargetHistogramPos + 24];
			shLocalHistograms[smemLocalHistogramPos + 25] += shLocalHistograms[smemTargetHistogramPos + 25];
			shLocalHistograms[smemLocalHistogramPos + 26] += shLocalHistograms[smemTargetHistogramPos + 26];
			shLocalHistograms[smemLocalHistogramPos + 27] += shLocalHistograms[smemTargetHistogramPos + 27];
			shLocalHistograms[smemLocalHistogramPos + 28] += shLocalHistograms[smemTargetHistogramPos + 28];
			shLocalHistograms[smemLocalHistogramPos + 29] += shLocalHistograms[smemTargetHistogramPos + 29];
			shLocalHistograms[smemLocalHistogramPos + 30] += shLocalHistograms[smemTargetHistogramPos + 30];
			shLocalHistograms[smemLocalHistogramPos + 31] += shLocalHistograms[smemTargetHistogramPos + 31];
			shLocalHistograms[smemLocalHistogramPos + 32] += shLocalHistograms[smemTargetHistogramPos + 32];
			shLocalHistograms[smemLocalHistogramPos + 33] += shLocalHistograms[smemTargetHistogramPos + 33];
			shLocalHistograms[smemLocalHistogramPos + 34] += shLocalHistograms[smemTargetHistogramPos + 34];
			shLocalHistograms[smemLocalHistogramPos + 35] += shLocalHistograms[smemTargetHistogramPos + 35];
#else
			for (i=0; i<histogramSize; i++)
				shLocalHistograms[smemLocalHistogramPos + i] += shLocalHistograms[smemTargetHistogramPos + i];
#endif
		}

		__syncthreads();
	}

	if (columnId == 0)
	{
		//write result to gmem
#ifdef UNROLL_LOOPS
		blockHistograms[gmemWritePos + 0].x = shLocalHistograms[smemReadPos + 0];
		blockHistograms[gmemWritePos + 1].x = shLocalHistograms[smemReadPos + 1];
		blockHistograms[gmemWritePos + 2].x = shLocalHistograms[smemReadPos + 2];
		blockHistograms[gmemWritePos + 3].x = shLocalHistograms[smemReadPos + 3];
		blockHistograms[gmemWritePos + 4].x = shLocalHistograms[smemReadPos + 4];
		blockHistograms[gmemWritePos + 5].x = shLocalHistograms[smemReadPos + 5];
		blockHistograms[gmemWritePos + 6].x = shLocalHistograms[smemReadPos + 6];
		blockHistograms[gmemWritePos + 7].x = shLocalHistograms[smemReadPos + 7];
		blockHistograms[gmemWritePos + 8].x = shLocalHistograms[smemReadPos + 8];
#else
		for (i=0; i<noHistogramBins; i++)
			blockHistograms[gmemWritePos + i].x = shLocalHistograms[smemReadPos + i];
#endif
	}

	if (blockIdx.x == 10 && blockIdx.y == 8)
	{
		int asasa;
		asasa = 0;
		asasa++;
	}
}
