//__global__ void resizeFastBicubic3(float4 *outputFloat, float4* paddedRegisteredImage, int width, int height, float scale)
__kernel void resizeFastBicubic(__global float4 *outputFloat ,__global float4* paddedRegisteredImage, const int width , const int height , const float scale )
{
	//int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	//int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	int x =(get_group_id(0)*get_group_size(0)) +get_local_id(0);
	int y =(get_group_id(1)*get_group_size(1) )+ get_local_id(1); 	


	int i = (y*width)+ x;

	float u = x*scale;
	float v = y*scale;

	if (x < width && y < height)
	{
		float4 cF;

		if (scale == 1.0f)
			cF = paddedRegisteredImage[x + y * width];
		else
			cF = tex2D(tex2, u, v);

		outputFloat[i] = cF;
	}

}