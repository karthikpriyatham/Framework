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

float1 convolutionRow(float1 *data,int i) {
	float1 val = data[convKernelRadius-i];
	val.x *= d_Kernel[i];
	float1 zero; zero.x = 0;
	if(i>0) val.x += convolutionRow((data).x,i-1).x;
	else val.x +=zero.x;
	return val;
}

float1 convolutionColumn(float1 *data,int i) {
	float1 val = data[(convKernelRadius-i)*convColumnTileWidth];
	val.x *= d_Kernel[i];
	float1 zero; zero.x = 0;
	if(i>0) val.x += convolutionRow((data).x,i-1).x;
	else val.x +=zero.x;
	return val;
}


int IMUL(int a,int b)
{
	return a*b;
}



__kernel void convolutionRowGPU1(__global float1 *d_Result,__global float1 *d_Data,const int dataW,const int dataH)

{
	float1 zero; zero.x = 0;

	//const int rowStart = IMUL(blockIdx.y, dataW);
		//rowStart = group_id.y * 1920
	const int rowStart = IMUL(get_group_id(1), dataW);


	//__shared__ float1 data[convKernelRadius + convRowTileWidth + convKernelRadius];
	__local float4 data[convKernelRadius + convRowTileWidth + convKernelRadius];
	

	//const int tileStart = IMUL(blockIdx.x, convRowTileWidth);
	const int tileStart = IMUL(get_group_id(0), convRowTileWidth);

	const int tileEnd = tileStart + convRowTileWidth - 1;
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
		float1 sum = convolutionRow(data + smemPos,2 * convKernelRadius);
		d_Result[rowStart + writePos] = sum;
	}


}
