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

float4 convolutionRow(float4 *data) {
	float4 val = data[convKernelRadius-i];
	val.x *= d_Kernel[i]; val.y *= d_Kernel[i];
	val.z *= d_Kernel[i]; val.w *= d_Kernel[i];
	float4 val2;
	float4 zero; zero.x = 0; zero.y = 0; zero.z = 0; zero.w = 0;
	if(i>0) val2  = convolutionRow(data,i-1);
	else val2 = zero;

	val.x += val2.x; val.y += val2.y;
	val.z += val2.z; val.w += val2.w;
	return val;

}


int IMUL(int a,int b)
{
	return a*b;
}



//onvolutionRowGPU4<<<blockGridRows, threadBlockRows>>>(convBuffer4, inputImage, convWidth, convHeight);
//__global__ void convolutionRowGPU4(float4 *d_Result, float4 *d_Data, int dataW, int dataH)  //1920 1440
__kernel void convolutionRowGPU4(__global float4 *d_Result,__global float4 *d_Data,const int dataW, const int dataH)

{
	float4 zero; zero.x = 0; zero.y = 0; zero.z = 0; zero.w = 0;

	//const int rowStart = IMUL(blockIdx.y, dataW);
	  const int rowStart = IMUL(get_group_id(1), dataW);
	  
	
	//__shared__ float4 data[convKernelRadius + convRowTileWidth + convKernelRadius];
	__local float4 data[convKernelRadius + convRowTileWidth + convKernelRadius];


	//const int tileStart = IMUL(blockIdx.x, convRowTileWidth);	block_id.x * 128
	  const int tileStart = IMUL(get_group_id(0),convRowTileWidth);	
	
	const int tileEnd = tileStart + convRowTileWidth - 1;	//
	const int apronStart = tileStart - convKernelRadius;
	const int apronEnd = tileEnd + convKernelRadius;

	const int tileEndClamped = min(tileEnd, dataW - 1);
	const int apronStartClamped = max(apronStart, 0);
	const int apronEndClamped = min(apronEnd, dataW - 1);

	const int apronStartAligned = tileStart - convKernelRadiusAligned;

	//const int loadPos = apronStartAligned + threadIdx.x;
	const int loadPos = apronStartAligned + get_local_id(0);


	if(loadPos >= apronStart)
	{
		const int smemPos = loadPos - apronStart;
		data[smemPos] = ((loadPos >= apronStartClamped) && (loadPos <= apronEndClamped)) ? d_Data[rowStart + loadPos] : zero;
	}

	//__syncthreads();
	barrier();
	
	//const int writePos = tileStart + threadIdx.x;
	const int writePos = tileStart + get_local_id(0);

	if(writePos <= tileEndClamped)
	{
		const int smemPos = writePos - apronStart;
		float4 sum = convolutionRow(data + smemPos,2 * convKernelRadius);
		d_Result[rowStart + writePos] = sum;
	}


}