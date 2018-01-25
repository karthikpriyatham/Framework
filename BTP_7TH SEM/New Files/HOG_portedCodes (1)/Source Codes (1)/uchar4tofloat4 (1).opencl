
__kernel void uchar4tofloat4(__global uchar4 *inputImage,__global  float4 *outputImage,const int width,const int height)
{
	//int offsetX = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetX = get_global_id(0);
	
	//int offsetY = blockIdx.y * blockDim.y + threadIdx.y;
	int offsetY = get_global_id(1);

	if (offsetX < width && offsetY < height)
	{
		//int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
		//int offset = offsetBlock + threadIdx.x + threadIdx.y * width;
	
		int offset = offsetX + offsetY*width;

		uchar4 pixel = inputImage[offset];
		float4 pixelf;
		pixelf.x = pixel.x; pixelf.y = pixel.y;
		pixelf.z = pixel.z; pixelf.w = pixel.w;

		outputImage[offset] = pixelf;

	}


}
