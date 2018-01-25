
__kernel void float4toUchar4(__global float4 *inputImage,__global uchar4 *outputImage,const int width,const int height)
{
	//int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
	//int offset = offsetBlock + threadIdx.x + threadIdx.y * width;
	
	int offset = get_global_id(0) + get_global_id(1)*width;

	float4 pixelf = inputImage[offset];
	uchar4 pixel;
	pixel.x = (unsigned char) pixelf.x; pixel.y = (unsigned char) pixelf.y;
	pixel.z = (unsigned char) pixelf.z; pixel.w = (unsigned char) pixelf.w;

	outputImage[offset] = pixel;
}