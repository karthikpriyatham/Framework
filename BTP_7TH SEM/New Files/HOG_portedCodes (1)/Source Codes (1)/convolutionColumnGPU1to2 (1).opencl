#define convKernelRadius 1
#define convKernelWidth (2 * convKernelRadius + 1)
__global __constant float d_Kernel[convKernelWidth];  //size 3
float *h_Kernel; 	//1.0f ,0f,-1.0f

#define convRowTileWidth 128
#define convKernelRadiusAligned 16

#define convColumnTileWidth 16
#define convColumnTileHeight 48
int convWidth;
int convHeight;
const int convKernelSize = convKernelWidth * sizeof(float); //3*float
bool convUseGrayscale;

float1 convolutionColumn(float1 *data,int i) {
	float1 val = data[(convKernelRadius-i)*convColumnTileWidth];
	val.x *= d_Kernel[i];
	float1 zero; zero.x = 0; 
	if(i>0 ) val.x +=convolutionColumn(data,i-1).x;
	val.x += zero.x;
	return val;
}

int IMUL(int a,int b)
{
	return a*b;
}

__kernel void convolutionColumnGPU1to2 (__global float2 *d_Result,__global float1 *d_Data,__global float1 *d_DataRow,const int dataW,const int dataH,const int smemStride,const int gmemStride)


{
	float1 rowValue;
	float1 zero; zero.x = 0;
	float2 result;

	//const int columnStart = IMUL(blockIdx.x, convColumnTileWidth) + threadIdx.x;
	const int columnStart = IMUL(blockIdx.x, convColumnTileWidth) + get_local_id(0);


//	__shared__ float1 data[convColumnTileWidth * (convKernelRadius + convColumnTileHeight + convKernelRadius)];

	__local float1 data[convColumnTileWidth * (convKernelRadius + convColumnTileHeight + convKernelRadius)];

	//const int tileStart = blockIdx.y*convColumnTileHeight;
	const int tileStart = get_group_id(1)*convColumnTileHeight;
		
	const int tileEnd = tileStart + convColumnTileHeight - 1;
	const int apronStart = tileStart - convKernelRadius;
	const int apronEnd = tileEnd   + convKernelRadius;

	const int tileEndClamped = min(tileEnd, dataH - 1);
	const int apronStartClamped = max(apronStart, 0);
	const int apronEndClamped = min(apronEnd, dataH - 1);

	//int smemPos = IMUL(threadIdx.y, convColumnTileWidth) + threadIdx.x;
	//int gmemPos = IMUL(apronStart + threadIdx.y, dataW) + columnStart;
	
	int smemPos = IMUL(get_local_id(1), convColumnTileWidth) + get_local_id(0);
	int gmemPos = IMUL(apronStart + get_local_id(1), dataW) + columnStart;



	//for(int y = apronStart + threadIdx.y; y <= apronEnd; y += blockDim.y)
	for(int y = apronStart + get_local_id(1); y <= apronEnd; y += get_group_size(1))
	{
		data[smemPos] = ((y >= apronStartClamped) && (y <= apronEndClamped)) ?  d_Data[gmemPos] : zero;
		smemPos += smemStride;
		gmemPos += gmemStride;
	}

	//__syncthreads();
	  barrier();

	//smemPos = IMUL(threadIdx.y + convKernelRadius, convColumnTileWidth) + threadIdx.x;
	//gmemPos = IMUL(tileStart + threadIdx.y , dataW) + columnStart;

	smemPos = IMUL(get_local_id(1)+convKernelRadius, convColumnTileWidth) + get_local_id(0);
	gmemPos = IMUL(tileStart + get_local_id(1), dataW) + columnStart;


	//for(int y = tileStart + threadIdx.y; y <= tileEndClamped; y += blockDim.y)
	for(int y = tileStart + get_local_id(1); y <= tileEndClamped; y += get_group_size(1))	
	{
		float1 sum = convolutionColumn(data + smemPos,2*convKernelRadius);
		rowValue = d_DataRow[gmemPos];

		result.x = sqrtf(sum.x * sum.x + rowValue.x * rowValue.x);
		result.y = atan2f(sum.x, rowValue.x) * RADTODEG;

		d_Result[gmemPos] = result;
		smemPos += smemStride;
		gmemPos += gmemStride;

	}

}
