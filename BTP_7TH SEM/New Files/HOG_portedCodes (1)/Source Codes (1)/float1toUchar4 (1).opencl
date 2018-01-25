__kernel void float1toUchar4(__global float1 *inputImage,__global uchar4 *outputImage,const int width, const int height)
{
	//int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
	//int offset = offsetBlock + threadIdx.x + threadIdx.y * width;
	
	int offset = get_global_id(0) + get_global_id(1)*width;

	float1 pixelf = inputImage[offset];
	uchar4 pixel;
	pixel.x = (unsigned char) pixelf.x; pixel.y = (unsigned char) pixelf.x;
	pixel.z = (unsigned char) pixelf.x; pixel.w = (unsigned char) pixelf.x;

	outputImage[offset] = pixel;
}