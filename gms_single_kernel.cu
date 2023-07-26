__global__ void first_jjb(float *bias,float *in_nu,float *in_w,float *out_nu,int model_num,
int in_fm,int out_fm,int str,int pad,int ker,int ker_channel,bool b,bool relu)
{
	int num_out = blockIdx.x;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

    int row = ((blockDim.x*row_out_block)+(row_out_thread));
    int col = ((blockDim.y*col_out_block)+(col_out_thread));  

	int out_position = (out_fm*out_fm*num_out)
					 + (out_fm*row)
					 + col;
	//Stride
    int x_str = 0, y_str = 0;
    x_str = 3*(row*str-pad)*in_fm;
    x_str = x_str < 0 ? 0 : x_str;
    y_str = 3*(col*str-pad);
    y_str = y_str < 0 ? 0 : y_str;

	//Padding
	int x_pad = 0, y_pad = 0;
	int loopr = ker, loopc = ker;

	//Upper
	if(row*str < pad){
		x_pad = pad - row*str;
		loopr = ker - x_pad;
	}
	//Bottom
	if(row >= out_fm - pad){
		loopr = in_fm - x_str/(3*in_fm);
	}
	//Left
	if(col*str < pad){
		y_pad = pad - col*str;
		loopc = ker - y_pad;
	}
	//Right
	if(col >= out_fm - pad){
		loopc = in_fm -  y_str/3;
	}

	float product[5] = {0.0};
	for(int l = 0; l < model_num; l++){
		for(int i = 0; i < loopr; i++){
			for(int j = 0; j < loopc; j++){
				for(int k = 0; k < ker_channel; k++){
					product[l] += in_nu[i*in_fm*ker_channel + j*ker_channel + k + x_str + y_str] 
							*in_w[num_out*ker*ker*ker_channel + i*ker + j + k*ker*ker + x_pad*ker + y_pad];
				}
			}
		}
		if(loopc > 0 && loopr > 0){
			if(b == true)
				product[l] += bias[num_out];

			//ReLU
			if(relu == true){
				if(product[l] < 0)
					product[l] = 0;
			}
			out_nu[out_position] = product[l];
		}
	}
}

/* Convolution */
__global__ void conv_jjb(float *bias,float *in_nu,float *in_w,float *out_nu,int model_num,
int in_fm,int out_fm,int str,int pad,int ker,int ker_channel,bool b,bool relu)
{
	int num_out = blockIdx.x;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

    int row = ((blockDim.x*row_out_block)+(row_out_thread));
    int col = ((blockDim.y*col_out_block)+(col_out_thread));  

	int out_position = (out_fm*out_fm*num_out) + (out_fm*row) + col;

    //Stride
    int x_str = 0, y_str = 0;
    x_str = (row*str-pad)*in_fm;
    x_str = x_str < 0 ? 0 : x_str;
    y_str = col*str-pad;
    y_str = y_str < 0 ? 0 : y_str;

	//Padding
	int x_pad = 0, y_pad = 0;
	int loopr = ker, loopc = ker;

	//Upper
	if(row*str < pad){
		x_pad = pad - row*str;
		loopr = ker - x_pad;
	}
	//Bottom
	if(row >= out_fm - pad){
		loopr = in_fm - x_str/in_fm;
	}
	//Left
	if(col*str < pad){
		y_pad = pad - col*str;
		loopc = ker - y_pad;
	}
	//Right
	if(col >= out_fm - pad){
		loopc = in_fm -  y_str;
	}

	float product[5] = {0.0};
	for(int l = 0; l < model_num; l++){
		for(int i = 0; i < ker_channel; i++){
			for(int j = 0; j < loopr; j++){
				for(int k = 0; k < loopc; k++){
					product[l] += in_nu[in_fm*in_fm*i + in_fm*j + k + x_str + y_str] 
							*in_w[num_out*ker_channel*ker*ker + i*ker*ker + j*ker + k + x_pad*ker + y_pad];
				}
			}
		}
		if(loopc > 0 && loopr > 0){
			if(b == true)
				product[l] += bias[num_out];

			//ReLU
			if(relu == true){
				if(product[l] < 0) 
					product[l] = 0;
			}
			out_nu[out_position] = product[l];
		}
	}
}

__global__ void gms_conv(float *bias,float *in_nu,float *in_w,float *out_nu,int model_num,
	int in_fm, int in_channel, 
	int out_fm, int out_channel, 
	int ker, int str, int pad, bool b, bool relu,
	int numOps)
{
	int block_index = blockDim.x * threadIdx.y + threadIdx.x; // block 내에서의 index
	int out_index = blockIdx.x * blockDim.x * blockDim.y + block_index;
	int offset = gridDim.x * blockDim.x * blockDim.y;

	int out_i, out_j, out_k; 	// Alexnet의 output에서의 (i, j, k) 위치
									// i: output kernel
									// j: output row
									// k: output column
	
	int num_elements = out_fm*out_fm*out_channel;

	for (int loops = 0; loops < numOps; loops++)
	{
		if (out_index < num_elements)
		{
			// out_index가 alexnet의 output에서의 좌표가 어디인지 (i, j, k)로 계산하기
			out_i = out_index / (out_fm*out_fm);
			out_j = (out_index % (out_fm*out_fm)) / out_fm;
			out_k = (out_index % (out_fm*out_fm)) % out_fm;

			// a_out(i, j, k)를 계산하기 위해 필요한 input data의 범위 찾기
			int in_k_min = out_k * str - pad;
			int in_k_max = in_k_min + ker - 1;

			int in_j_min = out_j * str - pad;
			int in_j_max = in_j_min + ker - 1;

			bool isElement = false;

			float product[5] = {0.0};

			for (int num = 0; num < model_num; num++)
			{
				
				for (int l = 0; l < in_channel; l++)
				{
					for (int j = in_j_min, ker_j = 0; j <= in_j_max; j++, ker_j++)
					{
						for (int k = in_k_min, ker_k = 0; k <= in_k_max; k++, ker_k++)
						{
							if (j >= 0 && k >= 0 && j < in_fm && k < in_fm)
							{
								product[num] += in_nu[(in_fm*in_fm) * l + (in_fm) * j + k]
												* in_w[(in_channel*ker*ker) * out_i
															+ (ker*ker) * l
															+ (ker) * ker_j
															+ ker_k];

								isElement = true;
							}
						}
					}
				}

				if (isElement)
				{
					if (b)
						product[num] += bias[out_i];

					if (relu && product[num] < 0)
						product[num] = 0;

					out_nu[out_index] = product[num];
				}
			}
		}

		out_index += offset;
	}
}

// __global__ void conv_jjb1(float *bias,float *in_nu,float *in_w,float *out_nu,int model_num,
// int in_fm,int out_fm,int str,int pad,int ker,int ker_channel,bool b,bool relu)
// {
// 	int num_out = blockIdx.x+64;
// 	int row_out_block = blockIdx.y;
// 	int col_out_block = blockIdx.z;
// 	int row_out_thread = threadIdx.x;
// 	int col_out_thread = threadIdx.y;

//     int row = ((blockDim.x*row_out_block)+(row_out_thread));
//     int col = ((blockDim.y*col_out_block)+(col_out_thread));  

// 	int out_position = (out_fm*out_fm*num_out) + (out_fm*row) + col;

//     //Stride
//     int x_str = 0, y_str = 0;
//     x_str = (row*str-pad)*in_fm;
//     x_str = x_str < 0 ? 0 : x_str;
//     y_str = col*str-pad;
//     y_str = y_str < 0 ? 0 : y_str;

// 	//Padding
// 	int x_pad = 0, y_pad = 0;
// 	int loopr = ker, loopc = ker;

// 	//Upper
// 	if(row*str < pad){
// 		x_pad = pad - row*str;
// 		loopr = ker - x_pad;
// 	}
// 	//Bottom
// 	if(row >= out_fm - pad){
// 		loopr = in_fm - x_str/in_fm;
// 	}
// 	//Left
// 	if(col*str < pad){
// 		y_pad = pad - col*str;
// 		loopc = ker - y_pad;
// 	}
// 	//Right
// 	if(col >= out_fm - pad){
// 		loopc = in_fm -  y_str;
// 	}

// 	float product[5] = {0.0};
// 	for(int l = 0; l < model_num; l++){
// 		for(int i = 0; i < ker_channel; i++){
// 			for(int j = 0; j < loopr; j++){
// 				for(int k = 0; k < loopc; k++){
// 					product[l] += in_nu[in_fm*in_fm*i + in_fm*j + k + x_str + y_str] 
// 							*in_w[num_out*ker_channel*ker*ker + i*ker*ker + j*ker + k + x_pad*ker + y_pad];
// 				}
// 			}
// 		}
// 		if(loopc > 0 && loopr > 0){
// 			if(b == true)
// 				product[l] += bias[num_out];

// 			//ReLU
// 			if(relu == true){
// 				if(product[l] < 0) 
// 					product[l] = 0;
// 			}
// 			out_nu[out_position] = product[l];
// 		}
// 	}
// }

/* Local Response Normalization */
__global__ void norm_jjb(float *in_nu,float *out_nu,int model_num,
float alpha,float beta,int local_size,int out_fm)
{
	int num_out = blockIdx.x;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

    int row = ((blockDim.x*row_out_block)+(row_out_thread));
    int col = ((blockDim.y*col_out_block)+(col_out_thread));  

	int out_position = (out_fm*out_fm*num_out) + (out_fm*row) + col;

    int input_position = (out_fm*row) + col;

    int nStart = 0, nEnd = 0;
    nStart=(num_out-2) > 1 ? (num_out-2) : 1 ;
    nEnd=(num_out+2) < gridDim.x ? (num_out+2) : gridDim.x ;

    float sum[5] = {0.0};
    float result[5] = {0.0};  
	for(int i = 0; i < model_num; i++){
		for(int j = (nStart-1); j < (nEnd-1); j++){
			sum[i] += pow((in_nu[j*out_fm*out_fm + input_position]),2);
		}
		result[i] = (in_nu[out_position]) / (pow( 1 + ((alpha/local_size) * sum[i]),beta));
		sum[i] = 0.0;
		out_nu[out_position] = result[i];
	}
}

/* Maxpooling */
__global__ void max_jjb(float *in_nu,float *out_nu,int model_num,
int in_fm,int out_fm,int str,int pad,int ker)
{
    int num_out = blockIdx.x;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

    int row = ((blockDim.x*row_out_block)+(row_out_thread));
    int col = ((blockDim.y*col_out_block)+(col_out_thread));  

	int out_position = (out_fm*out_fm*num_out) + (out_fm*row) + col;

    //Stride
    int x_str = 0, y_str = 0;
    x_str = (row*str-pad)*in_fm;
    x_str = x_str < 0 ? 0 : x_str;
    y_str = col*str-pad;
    y_str = y_str < 0 ? 0 : y_str;

	//Padding
	int loopr = ker, loopc = ker;

	//Upper
	if(row < pad){
		loopr = ker - pad;
	}
	//Bottom
	if(row >= out_fm - pad){
		loopr = in_fm - x_str/in_fm;
	}
	//Left
	if(col < pad){
		loopc = ker - pad;
	}
	//Right
	if(col >= out_fm - pad){
		loopc = in_fm -  y_str;
	}

    float max[5] = {0.0};
	for(int i = 0; i < model_num; i++){
		for(int j = 0; j < loopr; j++){
			for(int k = 0; k < loopc; k++){
				if(max[i] < (in_nu[num_out*in_fm*in_fm + j*in_fm + k + x_str + y_str]))
					max[i] = in_nu[num_out*in_fm*in_fm + j*in_fm + k + x_str + y_str];
			}
		}
    	out_nu[out_position] = max[i];
	}
}

/* Batch Normalization */
__global__ void batchnorm_jjb(float *in_nu,float *out_nu,int model_num,
float *mean,float *var,float *gamma,float *beta,int out_fm,bool relu)
{
    int num_out = blockIdx.x;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

    int row = ((blockDim.x*row_out_block)+(row_out_thread));
    int col = ((blockDim.y*col_out_block)+(col_out_thread));  

	int out_position = (out_fm*out_fm*num_out) + (out_fm*row) + col;

	float product[5] = {0.0};
	for(int i = 0; i < model_num; i++){
		product[i] = ((in_nu[out_position] - mean[num_out])/(sqrt(var[num_out] + 1e-5)))*gamma[num_out] + beta[num_out];
		//ReLU
		if(relu == true){
			if(product[i] < 0)
				product[i] = 0;
		}
		out_nu[out_position] = product[i];
	}
}

/* Basic Block(in resnet) */
__global__ void basic_block_jjb(float *in_nu1,float *in_nu2,float *out_nu,int model_num,
int out_fm,bool relu)
{
	int num_out = blockIdx.x;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

    int row = ((blockDim.x*row_out_block)+(row_out_thread));
    int col = ((blockDim.y*col_out_block)+(col_out_thread));

	int out_position = (out_fm*out_fm*num_out) + (out_fm*row) + col;

	float product[5] = {0.0};
	for(int i = 0; i < model_num; i++){
		product[i] = in_nu1[out_position] + in_nu2[out_position];
		//ReLU
		if(relu == true){
			if(product[i] < 0)
				product[i] = 0;
		}

		out_nu[out_position] = product[i];
	}
}

__global__ void globalavg_jjb(float *in_nu, float *out_nu,int model_num,int in_fm)
{
	int num_out = blockIdx.x;

	float sum[5] = {0.0};
	for(int k = 0; k < model_num; k++){
		for(int i = 0; i < in_fm; i++){
			for(int j = 0; j < in_fm; j++){
				sum[k] += in_nu[num_out*in_fm*in_fm + i*in_fm + j];
			}
		}
		out_nu[num_out] = sum[k]/(in_fm*in_fm);
	}
}

/* Fully Connected */
__global__ void fc_jjb(float *bias,float *in_nu,float *in_w,float *out_nu,int model_num,
int input, bool relu)
{
    int num_out = blockIdx.x;
	int weight = num_out * input;

	float result[5] = {0.0};
	for(int i = 0; i < model_num; i++){
		for(int j = 0; j < input; j++){
			result[i] += in_nu[j] * in_w[weight+j];
		}
		result[i] += bias[num_out];
		//ReLU
		if(relu == true){
			if(result[i] < 0)
				result[i] = 0;
		}
		out_nu[num_out] = result[i];
	}
}

