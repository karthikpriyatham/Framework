__kernel void float2toUchar4(__global  float2 *inputImage,__global uchar4 *outputImage,const int width,const int height,const int index)
{
	//int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
	//int offset = offsetBlock + threadIdx.x + threadIdx.y * width;
	
	int offset = get_global_id(0) + get_global_id(1)*width;



	float2 pixelf = inputImage[offset];
	float pixelfIndexed = (index == 0) ? pixelf.x : pixelf.y;

	uchar4 pixel;
	pixel.x = (unsigned char) abs(pixelfIndexed); pixel.y = (unsigned char) abs(pixelfIndexed);
	pixel.z = (unsigned char) abs(pixelfIndexed); pixel.w = (unsigned char) abs(pixelfIndexed);
	outputImage[offset] = pixel;
}