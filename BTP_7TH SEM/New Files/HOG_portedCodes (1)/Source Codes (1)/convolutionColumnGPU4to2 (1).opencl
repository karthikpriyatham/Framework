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

int IMUL(int a,int b)
{
	return a*b;
}

float4 convolutionColumn(float4 *data,int i) 
{

	float4 val = data[(convKernelRadius-i)*convColumnTileWidth];
	val.x *= d_Kernel[i]; val.y *= d_Kernel[i];
	val.z *= d_Kernel[i]; val.w *= d_Kernel[i];
	float4 val2;
	if(i>0) val2= convolutionColumn(data,i-1);
	else val2.x =val2.y = val2.z = val2.w =0;

	val.x += val2.x; val.y += val2.y;
	val.z += val2.z; val.w += val2.w;
	return val;

}




//__global__ void convolutionColumnGPU4to2 ( float2 *d_Result, float4 *d_Data, float4 *d_DataRow, int dataW, int dataH, int smemStride, int gmemStride)
//output , input , convbuffer, 1920 , 1440 , 16*8 , 1920*8 
__kernel void convolutionColumnGPU4to2 (__global float2 *d_Result,__global float4 *d_Data,__global float4 *d_DataRow,const int dataW,const int dataH,const int smemStride,const int gmemStride)
{
	//float3 max12, mag4;
	float3 mag1, mag2, mag3;
	float3 max34, magMax;
	float2 result;
	float4 rowValue;
	float4 zero; zero.x = 0; zero.y = 0; zero.z = 0; zero.w = 0;

	const int columnStart = IMUL(blockIdx.x, convColumnTileWidth) + threadIdx.x;

	//__shared__ float4 data[convColumnTileWidth * (convKernelRadius + convColumnTileHeight + convKernelRadius)];
     __local float1 data[convColumnTileWidth * (convKernelRadius + convColumnTileHeight + convKernelRadius)];

	
	const int tileStart = IMUL(blockIdx.y, convColumnTileHeight);
	const int tileEnd = tileStart + convColumnTileHeight - 1;
	const int apronStart = tileStart - convKernelRadius;
	const int apronEnd = tileEnd   + convKernelRadius;

	const int tileEndClamped = min(tileEnd, dataH - 1);
	const int apronStartClamped = max(apronStart, 0);
	const int apronEndClamped = min(apronEnd, dataH - 1);

	int smemPos = IMUL(threadIdx.y, convColumnTileWidth) + threadIdx.x;
	int gmemPos = IMUL(apronStart + threadIdx.y, dataW) + columnStart;

	for(int y = apronStart + threadIdx.y; y <= apronEnd; y += blockDim.y)
	{
		data[smemPos] = ((y >= apronStartClamped) && (y <= apronEndClamped)) ?  d_Data[gmemPos] : zero;
		smemPos += smemStride;
		gmemPos += gmemStride;
	}

	//__syncthreads();
	  barrier();

	smemPos = IMUL(threadIdx.y + convKernelRadius, convColumnTileWidth) + threadIdx.x;
	gmemPos = IMUL(tileStart + threadIdx.y , dataW) + columnStart;

	for(int y = tileStart + threadIdx.y; y <= tileEndClamped; y += blockDim.y)
	{
		float4 sum = convolutionColumn(data + smemPos,2 * convKernelRadius);
		rowValue = d_DataRow[gmemPos];

		mag1.x = sqrtf(sum.x * sum.x + rowValue.x * rowValue.x); mag1.y = sum.x; mag1.z = rowValue.x;
		mag2.x = sqrtf(sum.y * sum.y + rowValue.y * rowValue.y); mag2.y = sum.y; mag2.z = rowValue.y;
		mag3.x = sqrtf(sum.z * sum.z + rowValue.z * rowValue.z); mag3.y = sum.z; mag3.z = rowValue.z;

		max34 = (mag2.x > mag3.x) ? mag2 : mag3;
		magMax = (mag1.x > max34.x) ? mag1 : max34;

		result.x = magMax.x;
		result.y = atan2f(magMax.y, magMax.z);
		result.y = result.y * 180 / PI + 180;
		result.y = int(result.y) % 180; //TODO-> if semicerc

		d_Result[gmemPos] = result;
		smemPos += smemStride;
		gmemPos += gmemStride;
	
	}


}