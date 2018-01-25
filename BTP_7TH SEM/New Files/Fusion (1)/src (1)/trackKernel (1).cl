#define INVALID -2 

inline float sq(float r) {
	return r * r;
}

//removed M_data[4] and split it into parts
inline float3 Mat4TimeFloat3(float4 M_data_0,float4 M_data_1,float4 M_data_2,float4 M_data_3, float3 v) {
	return (float3)(
			dot((float3)(M_data_0.x,M_data_0.y,M_data_0.z), v)
					+ M_data_0.w,
			dot((float3)(M_data_1.x,M_data_1.y,M_data_1.z), v)
					+ M_data_1.w,
			dot((float3)(M_data_2.x,M_data_2.y,M_data_2.z), v)
					+M_data_2.w);
}

inline void setVolume(uint3 v_size, float3 v_dim, __global short2 *v_data, uint3 pos, float2 d) {
	v_data[pos.x + pos.y * v_size.x + pos.z * v_size.x * v_size.y] = (short2)(
			d.x * 32766.0f, d.y);
}

inline float3 posVolume(const uint3 v_size, const float3 v_dim, const __global short2 *v_data, const uint3 p) {
	return (float3)((p.x + 0.5f) * v_dim.x / v_size.x,
			(p.y + 0.5f) * v_dim.y / v_size.y,
			(p.z + 0.5f) * v_dim.z / v_size.z);
}

inline float2 getVolume(const uint3 v_size, const float3 v_dim, const __global short2* v_data, const uint3 pos) {
	const short2 d = v_data[pos.x + pos.y * v_size.x
			+ pos.z * v_size.x * v_size.y];
	return (float2)(d.x * 0.00003051944088f, d.y); //  / 32766.0f
}

inline float vs(const uint3 pos, const uint3 v_size, const float3 v_dim, const __global short2* v_data) {
	return v_data[pos.x + pos.y * v_size.x + pos.z * v_size.x * v_size.y].x;
}


inline float interp(const float3 pos, const uint3 v_size, const float3 v_dim, const __global short2 *v_data) {
	const float3 scaled_pos = (float3)((pos.x * v_size.x / v_dim.x) - 0.5f,
			(pos.y * v_size.y / v_dim.y) - 0.5f,
			(pos.z * v_size.z / v_dim.z) - 0.5f);

	float3 basef = (float3)(0);
	
	const int3 base = convert_int3(floor(scaled_pos));

	const float3 factor = (float3)(fract(scaled_pos, (float3 *) &basef));

	const int3 lower = max(base, (int3)(0));

	const int3 upper = min(base + (int3)(1), convert_int3(v_size) - (int3)(1));

	return (  (  ( vs  ( (uint3)(lower.x, lower.y, lower.z), v_size, v_dim, v_data) * (1 - factor.x)
			+ vs((uint3)(upper.x, lower.y, lower.z), v_size, v_dim, v_data) * factor.x)
			* (1 - factor.y)
			+ (vs((uint3)(lower.x, upper.y, lower.z), v_size, v_dim, v_data) * (1 - factor.x)
					+ vs((uint3)(upper.x, upper.y, lower.z), v_size, v_dim, v_data) * factor.x)
					* factor.y) * (1 - factor.z)
			+ ((vs((uint3)(lower.x, lower.y, upper.z), v_size, v_dim, v_data) * (1 - factor.x)
					+ vs((uint3)(upper.x, lower.y, upper.z), v_size, v_dim, v_data) * factor.x)
					* (1 - factor.y)
					+ (vs((uint3)(lower.x, upper.y, upper.z), v_size, v_dim, v_data)
							* (1 - factor.x)
							+ vs((uint3)(upper.x, upper.y, upper.z), v_size, v_dim, v_data)
									* factor.x) * factor.y) * factor.z)
			* 0.00003051944088f;
}

// Changed from Volume
inline float3 grad(float3 pos, const uint3 v_size, const float3 v_dim, const __global short2 *v_data) {
	const float3 scaled_pos = (float3)((pos.x * v_size.x / v_dim.x) - 0.5f,
			(pos.y * v_size.y / v_dim.y) - 0.5f,
			(pos.z * v_size.z / v_dim.z) - 0.5f);
	const int3 base = (int3)(floor(scaled_pos.x), floor(scaled_pos.y),
			floor(scaled_pos.z));
	const float3 basef = (float3)(0);
	const float3 factor = (float3) fract(scaled_pos, (float3 *) &basef);
	const int3 lower_lower = max(base - (int3)(1), (int3)(0));
	const int3 lower_upper = max(base, (int3)(0));
	const int3 upper_lower = min(base + (int3)(1),
			convert_int3(v_size) - (int3)(1));
	const int3 upper_upper = min(base + (int3)(2),
			convert_int3(v_size) - (int3)(1));
	const int3 lower = lower_upper;
	const int3 upper = upper_lower;

	float3 gradient;

	gradient.x = (((vs((uint3)(upper_lower.x, lower.y, lower.z), v_size, v_dim, v_data)
			- vs((uint3)(lower_lower.x, lower.y, lower.z),v_size, v_dim, v_data)) * (1 - factor.x)
			+ (vs((uint3)(upper_upper.x, lower.y, lower.z),  v_size, v_dim, v_data)
					- vs((uint3)(lower_upper.x, lower.y, lower.z),  v_size, v_dim, v_data))
					* factor.x) * (1 - factor.y)
			+ ((vs((uint3)(upper_lower.x, upper.y, lower.z),  v_size, v_dim, v_data)
					- vs((uint3)(lower_lower.x, upper.y, lower.z),  v_size, v_dim, v_data))
					* (1 - factor.x)
					+ (vs((uint3)(upper_upper.x, upper.y, lower.z),  v_size, v_dim, v_data)
							- vs((uint3)(lower_upper.x, upper.y, lower.z),  v_size, v_dim, v_data))
							* factor.x) * factor.y) * (1 - factor.z)
			+ (((vs((uint3)(upper_lower.x, lower.y, upper.z),  v_size, v_dim, v_data)
					- vs((uint3)(lower_lower.x, lower.y, upper.z),  v_size, v_dim, v_data))
					* (1 - factor.x)
					+ (vs((uint3)(upper_upper.x, lower.y, upper.z),  v_size, v_dim, v_data)
							- vs((uint3)(lower_upper.x, lower.y, upper.z),  v_size, v_dim, v_data))
							* factor.x) * (1 - factor.y)
					+ ((vs((uint3)(upper_lower.x, upper.y, upper.z),  v_size, v_dim, v_data)
							- vs((uint3)(lower_lower.x, upper.y, upper.z),  v_size, v_dim, v_data))
							* (1 - factor.x)
							+ (vs((uint3)(upper_upper.x, upper.y, upper.z),  v_size, v_dim, v_data)
									- vs(
											(uint3)(lower_upper.x, upper.y,
													upper.z),  v_size, v_dim, v_data)) * factor.x)
							* factor.y) * factor.z;

	gradient.y = (((vs((uint3)(lower.x, upper_lower.y, lower.z),  v_size, v_dim, v_data)
			- vs((uint3)(lower.x, lower_lower.y, lower.z),  v_size, v_dim, v_data)) * (1 - factor.x)
			+ (vs((uint3)(upper.x, upper_lower.y, lower.z),  v_size, v_dim, v_data)
					- vs((uint3)(upper.x, lower_lower.y, lower.z),  v_size, v_dim, v_data))
					* factor.x) * (1 - factor.y)
			+ ((vs((uint3)(lower.x, upper_upper.y, lower.z),  v_size, v_dim, v_data)
					- vs((uint3)(lower.x, lower_upper.y, lower.z),  v_size, v_dim, v_data))
					* (1 - factor.x)
					+ (vs((uint3)(upper.x, upper_upper.y, lower.z),  v_size, v_dim, v_data)
							- vs((uint3)(upper.x, lower_upper.y, lower.z),  v_size, v_dim, v_data))
							* factor.x) * factor.y) * (1 - factor.z)
			+ (((vs((uint3)(lower.x, upper_lower.y, upper.z),  v_size, v_dim, v_data)
					- vs((uint3)(lower.x, lower_lower.y, upper.z),  v_size, v_dim, v_data))
					* (1 - factor.x)
					+ (vs((uint3)(upper.x, upper_lower.y, upper.z),  v_size, v_dim, v_data)
							- vs((uint3)(upper.x, lower_lower.y, upper.z),  v_size, v_dim, v_data))
							* factor.x) * (1 - factor.y)
					+ ((vs((uint3)(lower.x, upper_upper.y, upper.z),  v_size, v_dim, v_data)
							- vs((uint3)(lower.x, lower_upper.y, upper.z),  v_size, v_dim, v_data))
							* (1 - factor.x)
							+ (vs((uint3)(upper.x, upper_upper.y, upper.z),  v_size, v_dim, v_data)
									- vs(
											(uint3)(upper.x, lower_upper.y,
													upper.z),  v_size, v_dim, v_data)) * factor.x)
							* factor.y) * factor.z;

	gradient.z = (((vs((uint3)(lower.x, lower.y, upper_lower.z),  v_size, v_dim, v_data)
			- vs((uint3)(lower.x, lower.y, lower_lower.z),  v_size, v_dim, v_data)) * (1 - factor.x)
			+ (vs((uint3)(upper.x, lower.y, upper_lower.z),  v_size, v_dim, v_data)
					- vs((uint3)(upper.x, lower.y, lower_lower.z),  v_size, v_dim, v_data))
					* factor.x) * (1 - factor.y)
			+ ((vs((uint3)(lower.x, upper.y, upper_lower.z),  v_size, v_dim, v_data)
					- vs((uint3)(lower.x, upper.y, lower_lower.z),  v_size, v_dim, v_data))
					* (1 - factor.x)
					+ (vs((uint3)(upper.x, upper.y, upper_lower.z),  v_size, v_dim, v_data)
							- vs((uint3)(upper.x, upper.y, lower_lower.z),  v_size, v_dim, v_data))
							* factor.x) * factor.y) * (1 - factor.z)
			+ (((vs((uint3)(lower.x, lower.y, upper_upper.z),  v_size, v_dim, v_data)
					- vs((uint3)(lower.x, lower.y, lower_upper.z),  v_size, v_dim, v_data))
					* (1 - factor.x)
					+ (vs((uint3)(upper.x, lower.y, upper_upper.z),  v_size, v_dim, v_data)
							- vs((uint3)(upper.x, lower.y, lower_upper.z),  v_size, v_dim, v_data))
							* factor.x) * (1 - factor.y)
					+ ((vs((uint3)(lower.x, upper.y, upper_upper.z),  v_size, v_dim, v_data)
							- vs((uint3)(lower.x, upper.y, lower_upper.z),  v_size, v_dim, v_data))
							* (1 - factor.x)
							+ (vs((uint3)(upper.x, upper.y, upper_upper.z),  v_size, v_dim, v_data)
									- vs(
											(uint3)(upper.x, upper.y,
													lower_upper.z),  v_size, v_dim, v_data))
									* factor.x) * factor.y) * factor.z;

	return gradient
			* (float3)(v_dim.x / v_size.x, v_dim.y / v_size.y,
					v_dim.z / v_size.z) * (0.5f * 0.00003051944088f);
}


// Changed from Matrix4
inline float3 get_translation(const float4 view_data_0,const float4 view_data_1,const float4 view_data_2,const float4 view_data_3) {
	return (float3)(view_data_0.w, view_data_1.w, view_data_2.w);
}


// Changed from Matrix4
inline float3 myrotate(const float4 M_data_0,const float4 M_data_1,const float4 M_data_2,const float4 M_data_3,const float3 v) {
	return (float3)(dot((float3)(M_data_0.x, M_data_0.y, M_data_0.z), v),
			dot((float3)(M_data_1.x, M_data_1.y, M_data_1.z), v),
			dot((float3)(M_data_2.x, M_data_2.y, M_data_2.z), v));
}


// Changed from Volume, Matrix4
float4 raycast(const uint3 v_size, const float3 v_dim, const __global short2 *v_data, const uint2 pos, const float4 view_data_0,
		const float4 view_data_1,const float4 view_data_2,const float4 view_data_3,const float nearPlane, const float farPlane,
		const float step,const float largestep) {

	const float3 origin = get_translation(view_data_0,view_data_1,view_data_2,view_data_3);
	const float3 direction = myrotate(view_data_0,view_data_1,view_data_2,view_data_3, (float3)(pos.x, pos.y, 1.f));

	// intersect ray with a box
	//
	// www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
	// compute intersection of ray with all six bbox planes
	const float3 invR = (float3)(1.0f) / direction;
	const float3 tbot = (float3) - 1 * invR * origin;
	const float3 ttop = invR * (v_dim - origin);

	// re-order intersections to find smallest and largest on each axis
	const float3 tmin = fmin(ttop, tbot);
	const float3 tmax = fmax(ttop, tbot);

	// find the largest tmin and the smallest tmax
	const float largest_tmin = fmax(fmax(tmin.x, tmin.y), fmax(tmin.x, tmin.z));
	const float smallest_tmax = fmin(fmin(tmax.x, tmax.y),
			fmin(tmax.x, tmax.z));

	// check against near and far plane
	const float tnear = fmax(largest_tmin, nearPlane);
	const float tfar = fmin(smallest_tmax, farPlane);

	if (tnear < tfar) {
		// first walk with largesteps until we found a hit
		float t = tnear;
		float stepsize = largestep;
		float f_t = interp(origin + direction * t, v_size, v_dim, v_data);
		float f_tt = 0;
		if (f_t > 0) { // ups, if we were already in it, then don't render anything here
			for (; t < tfar; t += stepsize) {
				f_tt = interp(origin + direction * t, v_size, v_dim, v_data);
				if (f_tt < 0)                  // got it, jump out of inner loop
					break;
				if (f_tt < 0.8f)               // coming closer, reduce stepsize
					stepsize = step;
				f_t = f_tt;
			}
			if (f_tt < 0) {           // got it, calculate accurate intersection
				t = t + stepsize * f_tt / (f_t - f_tt);
				return (float4)(origin + direction * t, t);
			}
		}
	}

	return (float4)(0);
}


/****************** KERNEL *****************/

__kernel void trackKernel (

		__global int* output_result,	//0
		__global float* output_error,	//1
		__global float* output_J,  //global array of size 6*dataset 2 

		const uint2 outputSize,	//3
		__global const float * inVertex,// float3 4
		const uint2 inVertexSize,	//5
		__global const float * inNormal,// float3	//6
		const uint2 inNormalSize,	//7
		__global const float * refVertex,// float3	//8
		const uint2 refVertexSize,	//9
		__global const float * refNormal,// float3 10
		const uint2 refNormalSize,	//11

		const float4 Ttrack_data_0,	//12
		const float4 Ttrack_data_1,	//13
		const float4 Ttrack_data_2,	//14
		const float4 Ttrack_data_3,	//15

		const float4 view_data_0,	//16
		const float4 view_data_1,	//17
		const float4 view_data_2,	//18
		const float4 view_data_3,	//19

		const float dist_threshold,	//20
		const float normal_threshold	//21

						)

{

	const uint2 pixel = (uint2)(get_global_id(0),get_global_id(1));

	if(pixel.x >= inVertexSize.x || pixel.y >= inVertexSize.y ) {return;}

	float3 inNormalPixel = vload3(pixel.x + inNormalSize.x * pixel.y,inNormal);

	if(inNormalPixel.x == INVALID ) {
		output_result[pixel.x + outputSize.x * pixel.y] = -1;
		return;
	}

	float3 inVertexPixel = vload3(pixel.x + inVertexSize.x * pixel.y,inVertex);
	const float3 projectedVertex = Mat4TimeFloat3 (Ttrack_data_0 ,Ttrack_data_1, Ttrack_data_2,Ttrack_data_3, inVertexPixel);
	const float3 projectedPos = Mat4TimeFloat3 ( view_data_0 , view_data_1 , view_data_2 , view_data_3 , projectedVertex);
	const float2 projPixel = (float2) ( projectedPos.x / projectedPos.z + 0.5f, projectedPos.y / projectedPos.z + 0.5f);

	if(projPixel.x < 0 || projPixel.x > refVertexSize.x-1 || projPixel.y < 0 || projPixel.y > refVertexSize.y-1 ) {
		output_result[pixel.x + outputSize.x * pixel.y] = -2;
		return;
	}

	const uint2 refPixel = (uint2) (projPixel.x, projPixel.y);
	const float3 referenceNormal = vload3(refPixel.x + refNormalSize.x * refPixel.y,refNormal);

	if(referenceNormal.x == INVALID) {
		output_result[pixel.x + outputSize.x * pixel.y] = -3;
		return;
	}

	const float3 diff = vload3(refPixel.x + refVertexSize.x * refPixel.y,refVertex) - projectedVertex;
	const float3 projectedNormal = myrotate(Ttrack_data_0,Ttrack_data_1,Ttrack_data_2,Ttrack_data_3, inNormalPixel);

	if(length(diff) > dist_threshold ) {
		output_result[pixel.x + outputSize.x * pixel.y] = -4;
		return;
	}
	if(dot(projectedNormal, referenceNormal) < normal_threshold) {
		output_result[pixel.x + outputSize.x * pixel.y] = -5;
		return;
	}

	output_result[pixel.x + outputSize.x * pixel.y] = 1;
	output_error[pixel.x + outputSize.x * pixel.y]  = dot(referenceNormal, diff);

	//added this here

	long d = pixel.x +outputSize.x*pixel.y;
	long f= (d-1)>=0 ? 6*(d-1):0;
	
	__private float temp[6]={0};
	temp[0]=output_J[f];
	temp[1]=output_J[f+1];
	temp[2]=output_J[f+2];
	temp[3]=output_J[f+3];
	temp[4]=output_J[f+4];
	temp[5]=output_J[f+5]; 
	
	vstore3( referenceNormal,0,temp );

	output_J[f]=temp[0];
	output_J[f+1]=temp[1];
	output_J[f+2]=temp[2];
	output_J[f+3]=temp[3];
	output_J[f+4]=temp[4];
	output_J[f+5]=temp[5];


	vstore3( cross(projectedVertex, referenceNormal),1,temp);

	output_J[f]=temp[0];
	output_J[f+1]=temp[1];
	output_J[f+2]=temp[2];
	output_J[f+3]=temp[3];
	output_J[f+4]=temp[4];
	output_J[f+5]=temp[5];


}

