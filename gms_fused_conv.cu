__global__ void fused_two_conv_block(float *a_bias,float *a_in_w,float *r_in_w,
	float *a_in_nu,float *r_in_nu,
	float *a_out_nu,float *r_out_nu,
	int alex_num,int res_num,
	int a_in_fm, int a_in_channel, int a_out_fm, int a_out_channel, int a_ker, int a_str, int a_pad, bool a_relu,
	int r_in_fm, int r_in_channel, int r_out_fm, int r_out_channel, int r_ker, int r_str, int r_pad, bool r_relu,
	int alexnet_warp, int resnet_warp,
	int alexNumOps, int resNumOps,
	bool first)
{
	int block_index = blockDim.x * threadIdx.y + threadIdx.x; // block 내에서의 index

	if (block_index < alexnet_warp * 32)
	{
		int out_index = blockIdx.x * (blockDim.x * alexnet_warp) // output array에서의 index (1차원)
						+ block_index;

		int offset = gridDim.x * blockDim.x * alexnet_warp;

		int a_out_i, a_out_j, a_out_k; 	// Alexnet의 output에서의 (i, j, k) 위치
										// i: output kernel
										// j: output row
										// k: output column
		
		int a_num_elements = a_out_fm*a_out_fm*a_out_channel;

		for (int loops = 0; loops < alexNumOps; loops++)
		{
			if (out_index < a_num_elements)
			{
				// out_index가 alexnet의 output에서의 좌표가 어디인지 (i, j, k)로 계산하기
				a_out_i = out_index / (a_out_fm*a_out_fm);
				a_out_j = (out_index % (a_out_fm*a_out_fm)) / a_out_fm;
				a_out_k = (out_index % (a_out_fm*a_out_fm)) % a_out_fm;

				// a_out(i, j, k)를 계산하기 위해 필요한 input data의 범위 찾기
				int a_in_k_min = a_out_k * a_str - a_pad;
				int a_in_k_max = a_in_k_min + a_ker - 1;

				int a_in_j_min = a_out_j * a_str - a_pad;
				int a_in_j_max = a_in_j_min + a_ker - 1;

				bool isElement = false;

				float a_product[5] = {0.0};
				for (int num = 0; num < alex_num; num++)
				{
					for (int l = 0; l < a_in_channel; l++)
					{
						for (int j = a_in_j_min, ker_j = 0; j <= a_in_j_max; j++, ker_j++)
						{
							for (int k = a_in_k_min, ker_k = 0; k <= a_in_k_max; k++, ker_k++)
							{
								if (j >= 0 && k >= 0 && j < a_in_fm && k < a_in_fm)
								{
									int idx;

									if (first)	idx = (a_in_fm*a_in_channel) * j + (a_in_channel) * k + l;
									else		idx = (a_in_fm*a_in_fm) * l + (a_in_fm) * j	+ k;

									a_product[num] += a_in_nu[idx]
													* a_in_w[(a_in_channel*a_ker*a_ker) * a_out_i
																+ (a_ker*a_ker) * l
																+ (a_ker) * ker_j
																+ ker_k];

									isElement = true;
								}
							}
						}
					}
					if (isElement)
					{
						a_product[num] += a_bias[a_out_i];

						if (a_relu && a_product[num] < 0)
							a_product[num] = 0;
						a_out_nu[out_index] = a_product[num];
					}

				}
			}

			out_index += offset;
		}
	}
	else
	{
		int out_index = blockIdx.x * (blockDim.x * resnet_warp) // output array에서의 index (1차원)
						+ block_index - alexnet_warp * 32;

		int offset = gridDim.x * blockDim.x * resnet_warp;

		int r_out_i, r_out_j, r_out_k; 	// Alexnet의 output에서의 (i, j, k) 위치
										// i: output kernel
										// j: output row
										// k: output column
		
		int r_num_elements = r_out_fm*r_out_fm*r_out_channel;

		for (int loops = 0; loops < resNumOps; loops++)
		{
			if (out_index < r_num_elements)
			{
				// out_index가 alexnet의 output에서의 좌표가 어디인지 (i, j, k)로 계산하기
				r_out_i = out_index / (r_out_fm*r_out_fm);
				r_out_j = (out_index % (r_out_fm*r_out_fm)) / r_out_fm;
				r_out_k = (out_index % (r_out_fm*r_out_fm)) % r_out_fm;

				// r_out(i, j, k)를 계산하기 위해 필요한 input data의 범위 찾기
				int r_in_k_min = r_out_k * r_str - r_pad;
				int r_in_k_max = r_in_k_min + r_ker - 1;

				int r_in_j_min = r_out_j * r_str - r_pad;
				int r_in_j_max = r_in_j_min + r_ker - 1;

				bool isElement = false;

				float r_product[5] = {0.0};
				for (int num = 0; num < res_num; num++)
				{
					for (int l = 0; l < r_in_channel; l++)
					{
						for (int j = r_in_j_min, ker_j = 0; j <= r_in_j_max; j++, ker_j++)
						{
							for (int k = r_in_k_min, ker_k = 0; k <= r_in_k_max; k++, ker_k++)
							{
								if (j >= 0 && k >= 0 && j < r_in_fm && k < r_in_fm)
								{
									int idx;

									if (first)	idx = (r_in_fm*r_in_channel) * j + (r_in_channel) * k + l;
									else		idx = (r_in_fm*r_in_fm) * l + (r_in_fm) * j	+ k;

									r_product[num] += r_in_nu[idx]
													* r_in_w[(r_in_channel*r_ker*r_ker) * r_out_i
																+ (r_ker*r_ker) * l
																+ (r_ker) * ker_j
																+ ker_k];
									isElement = true;
								}
							}
						}
					}
					if (isElement)
					{
						if (r_relu && r_product[num] < 0)
							r_product[num] = 0;
						r_out_nu[out_index] = r_product[num];
					}

				}
			}

			out_index += offset;
		}
	}
}
/*
__global__ void fused_two_conv_block_sh(float *a_bias,float *a_in_w,float *r_in_w,
	float *a_in_nu,float *r_in_nu,
	float *a_out_nu,float *r_out_nu,
	int alex_num,int res_num,
	int a_in_fm, int a_in_channel, int a_out_fm, int a_out_channel, int a_ker, int a_str, int a_pad, bool a_relu,
	int r_in_fm, int r_in_channel, int r_out_fm, int r_out_channel, int r_ker, int r_str, int r_pad, bool r_relu,
	int alexnet_warp, int resnet_warp,
	int alexNumOps, int resNumOps,
	int alexNumOps_sh, int resNumOps_sh,
	bool first)
{
	int block_index = blockDim.x * threadIdx.y + threadIdx.x; // block 내에서의 index

	__shared__ float a_in_w_sh[4800];
	// __shared__ float r_in_w_sh[576];

	if (block_index < alexnet_warp * 32)
	{
		int out_index = blockIdx.x * (blockDim.x * alexnet_warp) // output array에서의 index (1차원)
						+ block_index;

		int offset = gridDim.x * blockDim.x * alexnet_warp;

		int a_out_i, a_out_j, a_out_k; 	// Alexnet의 output에서의 (i, j, k) 위치
										// i: output kernel
										// j: output row
										// k: output column
		
		int a_num_elements = a_out_fm*a_out_fm*a_out_channel;

		for (int loops = 0; loops < alexNumOps; loops++)
		{
			int a_ker_i_start = (out_index - block_index) / (a_out_fm*a_out_fm);
			int a_ker_i_end = (out_index - block_index + 32 * alexnet_warp - 1) / (a_out_fm*a_out_fm);

			int block_index_sh = block_index;

			for (int loops2 = 0; loops2 < alexNumOps_sh; loops2++)
			{
				if (block_index_sh < (a_ker_i_end - a_ker_i_start + 1) * a_in_channel * a_ker * a_ker)
				{
					a_in_w_sh[block_index_sh] = a_in_w[a_ker_i_start*a_in_channel*a_ker*a_ker + block_index_sh];
				}
				block_index_sh += 32 * alexnet_warp;
			}

			__syncthreads();

			if (out_index < a_num_elements)
			{
				// out_index가 alexnet의 output에서의 좌표가 어디인지 (i, j, k)로 계산하기
				a_out_i = out_index / (a_out_fm*a_out_fm);
				a_out_j = (out_index % (a_out_fm*a_out_fm)) / a_out_fm;
				a_out_k = (out_index % (a_out_fm*a_out_fm)) % a_out_fm;

				// a_out(i, j, k)를 계산하기 위해 필요한 input data의 범위 찾기
				int a_in_k_min = a_out_k * a_str - a_pad;
				int a_in_k_max = a_in_k_min + a_ker - 1;

				int a_in_j_min = a_out_j * a_str - a_pad;
				int a_in_j_max = a_in_j_min + a_ker - 1;

				bool isElement = false;

				float a_product[5] = {0.0};
				for (int num = 0; num < alex_num; num++)
				{
					for (int l = 0; l < a_in_channel; l++)
					{
						for (int j = a_in_j_min, ker_j = 0; j <= a_in_j_max; j++, ker_j++)
						{
							for (int k = a_in_k_min, ker_k = 0; k <= a_in_k_max; k++, ker_k++)
							{
								if (j >= 0 && k >= 0 && j < a_in_fm && k < a_in_fm)
								{
									int idx;

									if (first)	idx = (a_in_fm*a_in_channel) * j + (a_in_channel) * k + l;
									else		idx = (a_in_fm*a_in_fm) * l + (a_in_fm) * j	+ k;

									a_product[num] += a_in_nu[idx]
														* a_in_w_sh[(a_in_channel*a_ker*a_ker) * (a_out_i - a_ker_i_start)
																+ (a_ker*a_ker) * l
																+ (a_ker) * ker_j
																+ ker_k];

									isElement = true;
								}
							}
						}
					}
					if (isElement)
					{
						a_product[num] += a_bias[a_out_i];

						if (a_relu)
							a_product[num] = max(0., a_product[num]);
						a_out_nu[out_index] = a_product[num];
					}

				}

				__syncthreads();
			}

			out_index += offset;
		}
	}
	else
	{
		// int out_index = blockIdx.x * (blockDim.x * resnet_warp) // output array에서의 index (1차원)
		// 				+ block_index - alexnet_warp * 32;

		// int offset = gridDim.x * blockDim.x * resnet_warp;

		// int r_out_i, r_out_j, r_out_k; 	// Alexnet의 output에서의 (i, j, k) 위치
		// 								// i: output kernel
		// 								// j: output row
		// 								// k: output column
		
		// int r_num_elements = r_out_fm*r_out_fm*r_out_channel;

		// for (int loops = 0; loops < resNumOps; loops++)
		// {
		// 	if (out_index < r_num_elements)
		// 	{
		// 		// out_index가 alexnet의 output에서의 좌표가 어디인지 (i, j, k)로 계산하기
		// 		r_out_i = out_index / (r_out_fm*r_out_fm);
		// 		r_out_j = (out_index % (r_out_fm*r_out_fm)) / r_out_fm;
		// 		r_out_k = (out_index % (r_out_fm*r_out_fm)) % r_out_fm;

		// 		// r_out(i, j, k)를 계산하기 위해 필요한 input data의 범위 찾기
		// 		int r_in_k_min = r_out_k * r_str - r_pad;
		// 		int r_in_k_max = r_in_k_min + r_ker - 1;

		// 		int r_in_j_min = r_out_j * r_str - r_pad;
		// 		int r_in_j_max = r_in_j_min + r_ker - 1;

		// 		bool isElement = false;

		// 		float r_product[5] = {0.0};
		// 		for (int num = 0; num < res_num; num++)
		// 		{
		// 			for (int l = 0; l < r_in_channel; l++)
		// 			{
		// 				for (int j = r_in_j_min, ker_j = 0; j <= r_in_j_max; j++, ker_j++)
		// 				{
		// 					for (int k = r_in_k_min, ker_k = 0; k <= r_in_k_max; k++, ker_k++)
		// 					{
		// 						if (j >= 0 && k >= 0 && j < r_in_fm && k < r_in_fm)
		// 						{
		// 							int idx;

		// 							if (first)	idx = (r_in_fm*r_in_channel) * j + (r_in_channel) * k + l;
		// 							else		idx = (r_in_fm*r_in_fm) * l + (r_in_fm) * j	+ k;

		// 							r_product[num] += r_in_nu[idx]
		// 											* r_in_w[(r_in_channel*r_ker*r_ker) * r_out_i
		// 														+ (r_ker*r_ker) * l
		// 														+ (r_ker) * ker_j
		// 														+ ker_k];
		// 							isElement = true;
		// 						}
		// 					}
		// 				}
		// 			}
		// 			if (isElement)
		// 			{
		// 				if (r_relu && r_product[num] < 0)
		// 					r_product[num] = 0;
		// 				r_out_nu[out_index] = r_product[num];
		// 			}

		// 		}
		// 	}

		// 	out_index += offset;
		// }
	}
}
*/

__global__ void fused_two_conv_thread(float *a_bias,float *a_in_w,float *r_in_w,
	float *a_in_nu,float *r_in_nu,
	float *a_out_nu,float *r_out_nu,
	int alex_num,int res_num,
	int a_in_fm, int a_in_channel, int a_out_fm, int a_out_channel, int a_ker, int a_str, int a_pad, bool a_relu,
	int r_in_fm, int r_in_channel, int r_out_fm, int r_out_channel, int r_ker, int r_str, int r_pad, bool r_relu,
	int alexnet_thread, int resnet_thread,
	int alexNumOps, int resNumOps)
{
	int block_index = blockDim.x * threadIdx.y + threadIdx.x; // block 내에서의 index

	if (block_index < alexnet_thread * 32)
	{
		int out_index = blockIdx.x * (blockDim.x * alexnet_thread) // output array에서의 index (1차원)
						+ block_index;

		int offset = gridDim.x * blockDim.x * alexnet_thread;

		int a_out_i, a_out_j, a_out_k; 	// Alexnet의 output에서의 (i, j, k) 위치
										// i: output kernel
										// j: output row
										// k: output column
		
		int a_num_elements = a_out_fm*a_out_fm*a_out_channel;

		for (int loops = 0; loops < alexNumOps; loops++)
		{
			if (out_index < a_num_elements)
			{
				// out_index가 alexnet의 output에서의 좌표가 어디인지 (i, j, k)로 계산하기
				a_out_i = out_index / (a_out_fm*a_out_fm);
				a_out_j = (out_index % (a_out_fm*a_out_fm)) / a_out_fm;
				a_out_k = (out_index % (a_out_fm*a_out_fm)) % a_out_fm;

				// a_out(i, j, k)를 계산하기 위해 필요한 input data의 범위 찾기
				int a_in_k_min = a_out_k * a_str - a_pad;
				int a_in_k_max = a_in_k_min + a_ker - 1;

				int a_in_j_min = a_out_j * a_str - a_pad;
				int a_in_j_max = a_in_j_min + a_ker - 1;

				bool isElement = false;

				float a_product[5] = {0.0};
				for (int num = 0; num < alex_num; num++)
				{
					for (int l = 0; l < a_in_channel; l++)
					{
						for (int j = a_in_j_min, ker_j = 0; j <= a_in_j_max; j++, ker_j++)
						{
							for (int k = a_in_k_min, ker_k = 0; k <= a_in_k_max; k++, ker_k++)
							{
								if (j >= 0 && k >= 0 && j < a_in_fm && k < a_in_fm)
								{
									a_product[num] += a_in_nu[(a_in_fm*a_in_fm) * l
																+ (a_in_fm) * j
																+ k]
													* a_in_w[(a_in_channel*a_ker*a_ker) * a_out_i
																+ (a_ker*a_ker) * l
																+ (a_ker) * ker_j
																+ ker_k];
									isElement = true;
								}
							}
						}
					}
					if (isElement)
					{
						a_product[num] += a_bias[a_out_i];

						if (a_relu && a_product[num] < 0)
							a_product[num] = 0;
						a_out_nu[out_index] = a_product[num];
					}

				}
			}

			out_index += offset;
		}
	}
	
	if (block_index < resnet_thread * 32)
	{
		int out_index = blockIdx.x * (blockDim.x * resnet_thread) // output array에서의 index (1차원)
						+ block_index;

		int offset = gridDim.x * blockDim.x * resnet_thread;

		int r_out_i, r_out_j, r_out_k; 	// Alexnet의 output에서의 (i, j, k) 위치
										// i: output kernel
										// j: output row
										// k: output column
		
		int r_num_elements = r_out_fm*r_out_fm*r_out_channel;

		for (int loops = 0; loops < resNumOps; loops++)
		{
			if (out_index < r_num_elements)
			{
				// out_index가 alexnet의 output에서의 좌표가 어디인지 (i, j, k)로 계산하기
				r_out_i = out_index / (r_out_fm*r_out_fm);
				r_out_j = (out_index % (r_out_fm*r_out_fm)) / r_out_fm;
				r_out_k = (out_index % (r_out_fm*r_out_fm)) % r_out_fm;

				// r_out(i, j, k)를 계산하기 위해 필요한 input data의 범위 찾기
				int r_in_k_min = r_out_k * r_str - r_pad;
				int r_in_k_max = r_in_k_min + r_ker - 1;

				int r_in_j_min = r_out_j * r_str - r_pad;
				int r_in_j_max = r_in_j_min + r_ker - 1;

				bool isElement = false;

				float r_product[5] = {0.0};
				for (int num = 0; num < res_num; num++)
				{
					for (int l = 0; l < r_in_channel; l++)
					{
						for (int j = r_in_j_min, ker_j = 0; j <= r_in_j_max; j++, ker_j++)
						{
							for (int k = r_in_k_min, ker_k = 0; k <= r_in_k_max; k++, ker_k++)
							{
								if (j >= 0 && k >= 0 && j < r_in_fm && k < r_in_fm)
								{
									r_product[num] += r_in_nu[(r_in_fm*r_in_fm) * l
																+ (r_in_fm) * j
																+ k]
													* r_in_w[(r_in_channel*r_ker*r_ker) * r_out_i
																+ (r_ker*r_ker) * l
																+ (r_ker) * ker_j
																+ ker_k];
									isElement = true;
								}
							}
						}
					}
					if (isElement)
					{
						if (r_relu && r_product[num] < 0)
							r_product[num] = 0;
						r_out_nu[out_index] = r_product[num];
					}

				}
			}

			out_index += offset;
		}
	}
}