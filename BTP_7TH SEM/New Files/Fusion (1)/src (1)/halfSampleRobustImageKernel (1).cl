/*

 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

/************** TYPES ***************/

#define INVALID -2 

typedef struct sVolume {
	uint3 size;
	float3 dim;
	__global short2 * data;
} Volume;

typedef struct sTrackData {
	int result;
	float error;
	float J[6];
} TrackData;

typedef struct sMatrix4 {
	float4 data[4];
} Matrix4;

/************** FUNCTIONS ***************/

inline float sq(float r) {
	return r * r;
}

inline float3 Mat4TimeFloat3(float4 M_data[], float3 v) {
	return (float3)(
			dot((float3)(M_data[0].x,M_data[0].y,M_data[0].z), v)
					+ M_data[0].w,
			dot((float3)(M_data[1].x,M_data[1].y,M_data[1].z), v)
					+ M_data[1].w,
			dot((float3)(M_data[2].x,M_data[2].y,M_data[2].z), v)
					+M_data[2].w);
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
inline float3 get_translation(const float4 view_data[]) {
	return (float3)(view_data[0].w, view_data[1].w, view_data[2].w);
}
// Changed from Matrix4
inline float3 myrotate(const float4 M_data[], const float3 v) {
	return (float3)(dot((float3)(M_data[0].x, M_data[0].y, M_data[0].z), v),
			dot((float3)(M_data[1].x, M_data[1].y, M_data[1].z), v),
			dot((float3)(M_data[2].x, M_data[2].y, M_data[2].z), v));
}
// Changed from Volume, Matrix4
float4 raycast(const uint3 v_size, const float3 v_dim, const __global short2 *v_data, const uint2 pos, const float4 view_data[],
		const float nearPlane, const float farPlane, const float step,
		const float largestep) {

	const float3 origin = get_translation(view_data);
	const float3 direction = myrotate(view_data, (float3)(pos.x, pos.y, 1.f));

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
/************** KERNELS ***************/

__kernel void renderNormalKernel( const __global uchar * in,
					__global float * out )
{

	const int posx = get_global_id(0);
	const int posy = get_global_id(1);
	const int sizex = get_global_size(0);

	float3 n;
	const uchar3 i = vload3(posx + sizex * posy,in);

	n.x = i.x;
	n.y = i.y;
	n.z = i.z;

	if(n.x == -2) {
		vstore3((float3) (0,0,0),posx + sizex * posy,out);
	} else {
		n = normalize(n);
		vstore3((float3) (n.x*128 + 128,n.y*128+128, n.z*128+128), posx + sizex * posy,out);
	}


}

__kernel void renderDepthKernel( __global uchar4 * out,
		__global float * depth,
		const float nearPlane,
		const float farPlane ) {

	const int posx = get_global_id(0);
	const int posy = get_global_id(1);
	const int sizex = get_global_size(0);
	float d= depth[posx + sizex * posy];
	if(d < nearPlane)
	vstore4((uchar4)(255, 255, 255, 0), posx + sizex * posy, (__global uchar*)out); // The forth value in uchar4 is padding for memory alignement and so it is for following uchar4 
	else {
		if(d > farPlane)
		vstore4((uchar4)(0, 0, 0, 0), posx + sizex * posy, (__global uchar*)out);
		else {
			float h =(d - nearPlane) / (farPlane - nearPlane);
			h *= 6.0f;
			const int sextant = (int)h;
			const float fract = h - sextant;
			const float mid1 = 0.25f + (0.5f*fract);
			const float mid2 = 0.75f - (0.5f*fract);
			switch (sextant)
			{
				case 0: vstore4((uchar4)(191, 255*mid1, 64, 0), posx + sizex * posy, (__global uchar*)out); break;
				case 1: vstore4((uchar4)(255*mid2, 191, 64, 0),posx + sizex * posy ,(__global uchar*)out); break;
				case 2: vstore4((uchar4)(64, 191, 255*mid1, 0),posx + sizex * posy ,(__global uchar*)out); break;
				case 3: vstore4((uchar4)(64, 255*mid2, 191, 0),posx + sizex * posy ,(__global uchar*)out); break;
				case 4: vstore4((uchar4)(255*mid1, 64, 191, 0),posx + sizex * posy ,(__global uchar*)out); break;
				case 5: vstore4((uchar4)(191, 64, 255*mid2, 0),posx + sizex * posy ,(__global uchar*)out); break;
			}
		}
	}
}


//changed track data * pointer
__kernel void renderTrackKernel( __global uchar3 * out,
		__global const int * data_result ) {

	const int posx = get_global_id(0);
	const int posy = get_global_id(1);
	const int sizex = get_global_size(0);

	switch(data_result[posx + sizex * posy]) {
		// The forth value in uchar4 is padding for memory alignement and so it is for following uchar4
		case  1: vstore4((uchar4)(128, 128, 128, 0), posx + sizex * posy, (__global uchar*)out); break; // ok	 GREY
		case -1: vstore4((uchar4)(000, 000, 000, 0), posx + sizex * posy, (__global uchar*)out); break; // no input BLACK
		case -2: vstore4((uchar4)(255, 000, 000, 0), posx + sizex * posy, (__global uchar*)out); break; // not in image RED
		case -3: vstore4((uchar4)(000, 255, 000, 0), posx + sizex * posy, (__global uchar*)out); break; // no correspondence GREEN
		case -4: vstore4((uchar4)(000, 000, 255, 0), posx + sizex * posy, (__global uchar*)out); break; // too far away BLUE
		case -5: vstore4((uchar4)(255, 255, 000, 0), posx + sizex * posy, (__global uchar*)out); break; // wrong normal YELLOW
		default: vstore4((uchar4)(255, 128, 128, 0), posx + sizex * posy, (__global uchar*)out); return;
	}
}
__kernel void halfSampleRobustImageKernel(__global float * out,
		__global const float * in,
		const uint2 inSize,
		const float e_d,
		const int r) {

	uint2 pixel = (uint2) (get_global_id(0),get_global_id(1));
	uint2 outSize = inSize / 2;

	const uint2 centerPixel = 2 * pixel;

	float sum = 0.0f;
	float t = 0.0f;
	const float center = in[centerPixel.x + centerPixel.y * inSize.x];
	for(int i = -r + 1; i <= r; ++i) {
		for(int j = -r + 1; j <= r; ++j) {
			int2 from = (int2)(clamp((int2)(centerPixel.x + j, centerPixel.y + i), (int2)(0), (int2)(inSize.x - 1, inSize.y - 1)));
			float current = in[from.x + from.y * inSize.x];
			if(fabs(current - center) < e_d) {
				sum += 1.0f;
				t += current;
			}
		}
	}
	out[pixel.x + pixel.y * outSize.x] = t / sum;

}

