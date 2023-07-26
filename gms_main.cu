#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "gms_memory.cu"
#include "gms_inference.cu"

int main(int argc, char* argv[])
{
	int is_inter_thread;

	is_inter_thread = atoi(argv[1]);

    float *Alex_Layer1_Neurons, * Alex_Layer2_Neurons, * Alex_Layer3_Neurons, * Alex_Layer4_Neurons; 
    float *Alex_Layer5_Neurons, * Alex_Layer6_Neurons, * Alex_Layer7_Neurons, * Alex_Layer8_Neurons; 
    float *Alex_Layer1_bias, * Alex_Layer2_bias, * Alex_Layer3_bias, * Alex_Layer4_bias; 
    float *Alex_Layer5_bias, * Alex_Layer6_bias, * Alex_Layer7_bias, * Alex_Layer8_bias; 
    float *Alex_Layer1_Weights, * Alex_Layer2_Weights, * Alex_Layer3_Weights, * Alex_Layer4_Weights; 
    float *Alex_Layer5_Weights, * Alex_Layer6_Weights, * Alex_Layer7_Weights, * Alex_Layer8_Weights; 
    float *Alex_Layer1_pool, * Alex_Layer2_pool, * Alex_Layer5_pool; 
    float *Alex_Layer1_norm, * Alex_Layer2_norm, * Alex_Result_Neurons; 
    float *Res_Layer1_Neurons, * Res_Layer2_Neurons, * Res_Layer3_Neurons, * Res_Layer4_Neurons; 
    float *Res_Layer5_Neurons, * Res_Layer6_Neurons, * Res_Layer7_Neurons, * Res_Layer8_Neurons; 
    float *Res_Layer9_Neurons, * Res_Layer10_Neurons, * Res_Layer11_Neurons, * Res_Layer12_Neurons; 
    float *Res_Layer13_Neurons, * Res_Layer14_Neurons, * Res_Layer15_Neurons, * Res_Layer16_Neurons; 
    float *Res_Layer17_Neurons, * Res_Layer18_Neurons; 
    float *Res_Layer1_Weights, * Res_Layer2_Weights, * Res_Layer3_Weights, * Res_Layer4_Weights; 
    float *Res_Layer5_Weights, * Res_Layer6_Weights, * Res_Layer7_Weights, * Res_Layer8_Weights; 
    float *Res_Layer9_Weights, * Res_Layer10_Weights, * Res_Layer11_Weights, * Res_Layer12_Weights; 
    float *Res_Layer13_Weights, * Res_Layer14_Weights, * Res_Layer15_Weights, * Res_Layer16_Weights; 
    float *Res_Layer17_Weights, * Res_Block3_Weights, * Res_Block4_Weights, * Res_Block5_Weights; 
    float *Res_Layer1_Gamma, * Res_Layer2_Gamma, * Res_Layer3_Gamma, * Res_Layer4_Gamma; 
    float *Res_Layer5_Gamma, * Res_Layer6_Gamma, * Res_Layer7_Gamma, * Res_Layer8_Gamma; 
    float *Res_Layer9_Gamma, * Res_Layer10_Gamma, * Res_Layer11_Gamma, * Res_Layer12_Gamma; 
    float *Res_Layer13_Gamma, * Res_Layer14_Gamma, * Res_Layer15_Gamma, * Res_Layer16_Gamma; 
    float *Res_Layer17_Gamma, * Res_Block3_Gamma, * Res_Block4_Gamma, * Res_Block5_Gamma; 
    float *Res_Layer1_Beta, * Res_Layer2_Beta, * Res_Layer3_Beta, * Res_Layer4_Beta; 
    float *Res_Layer5_Beta, * Res_Layer6_Beta, * Res_Layer7_Beta, * Res_Layer8_Beta; 
    float *Res_Layer9_Beta, * Res_Layer10_Beta, * Res_Layer11_Beta, * Res_Layer12_Beta; 
    float *Res_Layer13_Beta, * Res_Layer14_Beta, * Res_Layer15_Beta, * Res_Layer16_Beta; 
    float *Res_Layer17_Beta, * Res_Block3_Beta, * Res_Block4_Beta, * Res_Block5_Beta; 
    float *Res_mean1, * Res_mean2, * Res_mean3, * Res_mean4, * Res_mean5; 
    float *Res_mean6, * Res_mean7, * Res_mean8, * Res_mean9, * Res_mean10; 
    float *Res_mean11, * Res_mean12, * Res_mean13, * Res_mean14, * Res_mean15; 
    float *Res_mean16, * Res_mean17, * Res_Block3_mean, * Res_Block4_mean, * Res_Block5_mean; 
    float *Res_var1, * Res_var2, * Res_var3, * Res_var4, * Res_var5; 
    float *Res_var6, * Res_var7, * Res_var8, * Res_var9, * Res_var10; 
    float *Res_var11, * Res_var12, * Res_var13, * Res_var14, * Res_var15; 
    float *Res_var16, * Res_var17, * Res_Block3_var, * Res_Block4_var, * Res_Block5_var; 
    float *Res_FC_bias, * Res_FC_Weights; 
    float *Res_Layer3_basic, * Res_Layer5_basic, * Res_Layer7_basic, * Res_Layer9_basic; 
    float *Res_Layer11_basic, * Res_Layer13_basic, * Res_Layer15_basic, * Res_Layer17_basic; 
    float *Res_Block3_basic, * Res_Block4_basic, * Res_Block5_basic; 
    float *Res_Layer1_bn, * Res_Layer2_bn, * Res_Layer3_bn, * Res_Layer4_bn; 
    float *Res_Layer5_bn, * Res_Layer6_bn, * Res_Layer7_bn, * Res_Layer8_bn; 
    float *Res_Layer9_bn, * Res_Layer10_bn, * Res_Layer11_bn, * Res_Layer12_bn; 
    float *Res_Layer13_bn, * Res_Layer14_bn, * Res_Layer15_bn, * Res_Layer16_bn; 
    float *Res_Layer17_bn, * Res_Block3_bn, * Res_Block4_bn, * Res_Block5_bn; 
    float *Res_Layer1_pool, * Res_FC_Neurons, * Res_Result_Neurons;

    int alex_num = 2, res_num = 2;

	Alex_Res_host2gpu(&Alex_Layer1_Neurons,&Alex_Layer2_Neurons,&Alex_Layer3_Neurons,&Alex_Layer4_Neurons,
					&Alex_Layer5_Neurons,&Alex_Layer6_Neurons,&Alex_Layer7_Neurons,&Alex_Layer8_Neurons,
                    &Alex_Layer1_bias,&Alex_Layer2_bias,&Alex_Layer3_bias,&Alex_Layer4_bias,
                    &Alex_Layer5_bias,&Alex_Layer6_bias,&Alex_Layer7_bias,&Alex_Layer8_bias,
                    &Alex_Layer1_Weights,&Alex_Layer2_Weights,&Alex_Layer3_Weights,&Alex_Layer4_Weights,
                    &Alex_Layer5_Weights,&Alex_Layer6_Weights,&Alex_Layer7_Weights,&Alex_Layer8_Weights,
                    &Alex_Layer1_pool,&Alex_Layer2_pool,&Alex_Layer5_pool,
					&Alex_Layer1_norm,&Alex_Layer2_norm,&Alex_Result_Neurons,
					&Res_Layer1_Neurons,&Res_Layer2_Neurons,&Res_Layer3_Neurons,&Res_Layer4_Neurons,
					&Res_Layer5_Neurons,&Res_Layer6_Neurons,&Res_Layer7_Neurons,&Res_Layer8_Neurons,
					&Res_Layer9_Neurons,&Res_Layer10_Neurons,&Res_Layer11_Neurons,&Res_Layer12_Neurons,
					&Res_Layer13_Neurons,&Res_Layer14_Neurons,&Res_Layer15_Neurons,&Res_Layer16_Neurons,
					&Res_Layer17_Neurons,&Res_Layer18_Neurons,
                    &Res_Layer1_Weights,&Res_Layer2_Weights,&Res_Layer3_Weights,&Res_Layer4_Weights,
                    &Res_Layer5_Weights,&Res_Layer6_Weights,&Res_Layer7_Weights,&Res_Layer8_Weights,
                    &Res_Layer9_Weights,&Res_Layer10_Weights,&Res_Layer11_Weights,&Res_Layer12_Weights,
                    &Res_Layer13_Weights,&Res_Layer14_Weights,&Res_Layer15_Weights,&Res_Layer16_Weights,
                    &Res_Layer17_Weights,&Res_Block3_Weights,&Res_Block4_Weights,&Res_Block5_Weights,
                    &Res_Layer1_Gamma,&Res_Layer2_Gamma,&Res_Layer3_Gamma,&Res_Layer4_Gamma,
                    &Res_Layer5_Gamma,&Res_Layer6_Gamma,&Res_Layer7_Gamma,&Res_Layer8_Gamma,
                    &Res_Layer9_Gamma,&Res_Layer10_Gamma,&Res_Layer11_Gamma,&Res_Layer12_Gamma,
                    &Res_Layer13_Gamma,&Res_Layer14_Gamma,&Res_Layer15_Gamma,&Res_Layer16_Gamma,
                    &Res_Layer17_Gamma,&Res_Block3_Gamma,&Res_Block4_Gamma,&Res_Block5_Gamma,
                    &Res_Layer1_Beta,&Res_Layer2_Beta,&Res_Layer3_Beta,&Res_Layer4_Beta,
                    &Res_Layer5_Beta,&Res_Layer6_Beta,&Res_Layer7_Beta,&Res_Layer8_Beta,
                    &Res_Layer9_Beta,&Res_Layer10_Beta,&Res_Layer11_Beta,&Res_Layer12_Beta,
                    &Res_Layer13_Beta,&Res_Layer14_Beta,&Res_Layer15_Beta,&Res_Layer16_Beta,
                    &Res_Layer17_Beta,&Res_Block3_Beta,&Res_Block4_Beta,&Res_Block5_Beta,
                    &Res_mean1,&Res_mean2,&Res_mean3,&Res_mean4,&Res_mean5,
                    &Res_mean6,&Res_mean7,&Res_mean8,&Res_mean9,&Res_mean10,
                    &Res_mean11,&Res_mean12,&Res_mean13,&Res_mean14,&Res_mean15,
                    &Res_mean16,&Res_mean17,&Res_Block3_mean,&Res_Block4_mean,&Res_Block5_mean,
                    &Res_var1,&Res_var2,&Res_var3,&Res_var4,&Res_var5,
                    &Res_var6,&Res_var7,&Res_var8,&Res_var9,&Res_var10,
                    &Res_var11,&Res_var12,&Res_var13,&Res_var14,&Res_var15,
                    &Res_var16,&Res_var17,&Res_Block3_var,&Res_Block4_var,&Res_Block5_var,
                    &Res_FC_bias,&Res_FC_Weights,
					&Res_Layer3_basic,&Res_Layer5_basic,&Res_Layer7_basic,&Res_Layer9_basic,
					&Res_Layer11_basic,&Res_Layer13_basic,&Res_Layer15_basic,&Res_Layer17_basic,
					&Res_Block3_basic,&Res_Block4_basic,&Res_Block5_basic,
					&Res_Layer1_bn,&Res_Layer2_bn,&Res_Layer3_bn,&Res_Layer4_bn,
					&Res_Layer5_bn,&Res_Layer6_bn,&Res_Layer7_bn,&Res_Layer8_bn,
					&Res_Layer9_bn,&Res_Layer10_bn,&Res_Layer11_bn,&Res_Layer12_bn,
					&Res_Layer13_bn,&Res_Layer14_bn,&Res_Layer15_bn,&Res_Layer16_bn,
					&Res_Layer17_bn,&Res_Block3_bn,&Res_Block4_bn,&Res_Block5_bn,
					&Res_Layer1_pool,&Res_FC_Neurons,&Res_Result_Neurons);

	// Alex_Res_inference_block(Alex_Layer1_Neurons,Alex_Layer2_Neurons,Alex_Layer3_Neurons,Alex_Layer4_Neurons,
	// 			Alex_Layer5_Neurons,Alex_Layer6_Neurons,Alex_Layer7_Neurons,Alex_Layer8_Neurons,
	// 			Alex_Layer1_bias,Alex_Layer2_bias,Alex_Layer3_bias,Alex_Layer4_bias,
	// 			Alex_Layer5_bias,Alex_Layer6_bias,Alex_Layer7_bias,Alex_Layer8_bias,
	// 			Alex_Layer1_Weights,Alex_Layer2_Weights,Alex_Layer3_Weights,Alex_Layer4_Weights,
	// 			Alex_Layer5_Weights, Alex_Layer6_Weights,Alex_Layer7_Weights,Alex_Layer8_Weights,
	// 			Alex_Layer1_pool,Alex_Layer2_pool,Alex_Layer5_pool,
	// 			Alex_Layer1_norm,Alex_Layer2_norm,Alex_Result_Neurons,
	// 			Res_Layer1_Neurons,Res_Layer2_Neurons,Res_Layer3_Neurons,Res_Layer4_Neurons,
	// 			Res_Layer5_Neurons,Res_Layer6_Neurons,Res_Layer7_Neurons,Res_Layer8_Neurons,
	// 			Res_Layer9_Neurons,Res_Layer10_Neurons,Res_Layer11_Neurons,Res_Layer12_Neurons,
	// 			Res_Layer13_Neurons,Res_Layer14_Neurons,Res_Layer15_Neurons,Res_Layer16_Neurons,
	// 			Res_Layer17_Neurons,Res_Layer18_Neurons,
	// 			Res_Layer1_Weights,Res_Layer2_Weights,Res_Layer3_Weights,Res_Layer4_Weights,
	// 			Res_Layer5_Weights,Res_Layer6_Weights,Res_Layer7_Weights,Res_Layer8_Weights,
	// 			Res_Layer9_Weights,Res_Layer10_Weights,Res_Layer11_Weights,Res_Layer12_Weights,
	// 			Res_Layer13_Weights,Res_Layer14_Weights,Res_Layer15_Weights,Res_Layer16_Weights,
	// 			Res_Layer17_Weights,Res_Block3_Weights,Res_Block4_Weights,Res_Block5_Weights,
	// 			Res_Layer1_Gamma,Res_Layer2_Gamma,Res_Layer3_Gamma,Res_Layer4_Gamma,
	// 			Res_Layer5_Gamma,Res_Layer6_Gamma,Res_Layer7_Gamma,Res_Layer8_Gamma,
	// 			Res_Layer9_Gamma,Res_Layer10_Gamma,Res_Layer11_Gamma,Res_Layer12_Gamma,
	// 			Res_Layer13_Gamma,Res_Layer14_Gamma,Res_Layer15_Gamma,Res_Layer16_Gamma,
	// 			Res_Layer17_Gamma,Res_Block3_Gamma,Res_Block4_Gamma,Res_Block5_Gamma,
	// 			Res_Layer1_Beta,Res_Layer2_Beta,Res_Layer3_Beta,Res_Layer4_Beta,
	// 			Res_Layer5_Beta,Res_Layer6_Beta,Res_Layer7_Beta,Res_Layer8_Beta,
	// 			Res_Layer9_Beta,Res_Layer10_Beta,Res_Layer11_Beta,Res_Layer12_Beta,
	// 			Res_Layer13_Beta,Res_Layer14_Beta,Res_Layer15_Beta,Res_Layer16_Beta,
	// 			Res_Layer17_Beta,Res_Block3_Beta,Res_Block4_Beta,Res_Block5_Beta,
	// 			Res_mean1,Res_mean2,Res_mean3,Res_mean4,Res_mean5,
	// 			Res_mean6,Res_mean7,Res_mean8,Res_mean9,Res_mean10,
	// 			Res_mean11,Res_mean12,Res_mean13,Res_mean14,Res_mean15,
	// 			Res_mean16,Res_mean17,Res_Block3_mean,Res_Block4_mean,Res_Block5_mean,
	// 			Res_var1,Res_var2,Res_var3,Res_var4,Res_var5,
	// 			Res_var6,Res_var7,Res_var8,Res_var9,Res_var10,
	// 			Res_var11,Res_var12,Res_var13,Res_var14,Res_var15,
	// 			Res_var16,Res_var17,Res_Block3_var,Res_Block4_var,Res_Block5_var,
	// 			Res_FC_bias,Res_FC_Weights,
	// 			Res_Layer3_basic,Res_Layer5_basic,Res_Layer7_basic,Res_Layer9_basic,
	// 			Res_Layer11_basic,Res_Layer13_basic,Res_Layer15_basic,Res_Layer17_basic,
	// 			Res_Block3_basic,Res_Block4_basic,Res_Block5_basic,
	// 			Res_Layer1_bn,Res_Layer2_bn,Res_Layer3_bn,Res_Layer4_bn,
	// 			Res_Layer5_bn,Res_Layer6_bn,Res_Layer7_bn,Res_Layer8_bn,
	// 			Res_Layer9_bn,Res_Layer10_bn,Res_Layer11_bn,Res_Layer12_bn,
	// 			Res_Layer13_bn,Res_Layer14_bn,Res_Layer15_bn,Res_Layer16_bn,
	// 			Res_Layer17_bn,Res_Block3_bn,Res_Block4_bn,Res_Block5_bn,
	// 			Res_Layer1_pool,Res_FC_Neurons,Res_Result_Neurons,
	// 			2, 2,
	// 			72, 15, 17);

	// return 0;

	if (is_inter_thread)
	{
		for (int alex = 1; alex < 33; alex++) for (int res = 1; res < 33; res++)
		{
			Alex_Res_inference_thread(Alex_Layer1_Neurons,Alex_Layer2_Neurons,Alex_Layer3_Neurons,Alex_Layer4_Neurons,
							Alex_Layer5_Neurons,Alex_Layer6_Neurons,Alex_Layer7_Neurons,Alex_Layer8_Neurons,
							Alex_Layer1_bias,Alex_Layer2_bias,Alex_Layer3_bias,Alex_Layer4_bias,
							Alex_Layer5_bias,Alex_Layer6_bias,Alex_Layer7_bias,Alex_Layer8_bias,
							Alex_Layer1_Weights,Alex_Layer2_Weights,Alex_Layer3_Weights,Alex_Layer4_Weights,
							Alex_Layer5_Weights, Alex_Layer6_Weights,Alex_Layer7_Weights,Alex_Layer8_Weights,
							Alex_Layer1_pool,Alex_Layer2_pool,Alex_Layer5_pool,
							Alex_Layer1_norm,Alex_Layer2_norm,Alex_Result_Neurons,
							Res_Layer1_Neurons,Res_Layer2_Neurons,Res_Layer3_Neurons,Res_Layer4_Neurons,
							Res_Layer5_Neurons,Res_Layer6_Neurons,Res_Layer7_Neurons,Res_Layer8_Neurons,
							Res_Layer9_Neurons,Res_Layer10_Neurons,Res_Layer11_Neurons,Res_Layer12_Neurons,
							Res_Layer13_Neurons,Res_Layer14_Neurons,Res_Layer15_Neurons,Res_Layer16_Neurons,
							Res_Layer17_Neurons,Res_Layer18_Neurons,
							Res_Layer1_Weights,Res_Layer2_Weights,Res_Layer3_Weights,Res_Layer4_Weights,
							Res_Layer5_Weights,Res_Layer6_Weights,Res_Layer7_Weights,Res_Layer8_Weights,
							Res_Layer9_Weights,Res_Layer10_Weights,Res_Layer11_Weights,Res_Layer12_Weights,
							Res_Layer13_Weights,Res_Layer14_Weights,Res_Layer15_Weights,Res_Layer16_Weights,
							Res_Layer17_Weights,Res_Block3_Weights,Res_Block4_Weights,Res_Block5_Weights,
							Res_Layer1_Gamma,Res_Layer2_Gamma,Res_Layer3_Gamma,Res_Layer4_Gamma,
							Res_Layer5_Gamma,Res_Layer6_Gamma,Res_Layer7_Gamma,Res_Layer8_Gamma,
							Res_Layer9_Gamma,Res_Layer10_Gamma,Res_Layer11_Gamma,Res_Layer12_Gamma,
							Res_Layer13_Gamma,Res_Layer14_Gamma,Res_Layer15_Gamma,Res_Layer16_Gamma,
							Res_Layer17_Gamma,Res_Block3_Gamma,Res_Block4_Gamma,Res_Block5_Gamma,
							Res_Layer1_Beta,Res_Layer2_Beta,Res_Layer3_Beta,Res_Layer4_Beta,
							Res_Layer5_Beta,Res_Layer6_Beta,Res_Layer7_Beta,Res_Layer8_Beta,
							Res_Layer9_Beta,Res_Layer10_Beta,Res_Layer11_Beta,Res_Layer12_Beta,
							Res_Layer13_Beta,Res_Layer14_Beta,Res_Layer15_Beta,Res_Layer16_Beta,
							Res_Layer17_Beta,Res_Block3_Beta,Res_Block4_Beta,Res_Block5_Beta,
							Res_mean1,Res_mean2,Res_mean3,Res_mean4,Res_mean5,
							Res_mean6,Res_mean7,Res_mean8,Res_mean9,Res_mean10,
							Res_mean11,Res_mean12,Res_mean13,Res_mean14,Res_mean15,
							Res_mean16,Res_mean17,Res_Block3_mean,Res_Block4_mean,Res_Block5_mean,
							Res_var1,Res_var2,Res_var3,Res_var4,Res_var5,
							Res_var6,Res_var7,Res_var8,Res_var9,Res_var10,
							Res_var11,Res_var12,Res_var13,Res_var14,Res_var15,
							Res_var16,Res_var17,Res_Block3_var,Res_Block4_var,Res_Block5_var,
							Res_FC_bias,Res_FC_Weights,
							Res_Layer3_basic,Res_Layer5_basic,Res_Layer7_basic,Res_Layer9_basic,
							Res_Layer11_basic,Res_Layer13_basic,Res_Layer15_basic,Res_Layer17_basic,
							Res_Block3_basic,Res_Block4_basic,Res_Block5_basic,
							Res_Layer1_bn,Res_Layer2_bn,Res_Layer3_bn,Res_Layer4_bn,
							Res_Layer5_bn,Res_Layer6_bn,Res_Layer7_bn,Res_Layer8_bn,
							Res_Layer9_bn,Res_Layer10_bn,Res_Layer11_bn,Res_Layer12_bn,
							Res_Layer13_bn,Res_Layer14_bn,Res_Layer15_bn,Res_Layer16_bn,
							Res_Layer17_bn,Res_Block3_bn,Res_Block4_bn,Res_Block5_bn,
							Res_Layer1_pool,Res_FC_Neurons,Res_Result_Neurons,
							alex_num, res_num,
							10, alex, res);
		}
	}
	else
	{
		for (int gridSize = 1; gridSize <= 24; gridSize += 1)
		{
			for (int b = 32; b >= 32; b--)
			{
				for (int i = 1; i < b; i++)
				{
					Alex_Res_inference_block(Alex_Layer1_Neurons,Alex_Layer2_Neurons,Alex_Layer3_Neurons,Alex_Layer4_Neurons,
								Alex_Layer5_Neurons,Alex_Layer6_Neurons,Alex_Layer7_Neurons,Alex_Layer8_Neurons,
								Alex_Layer1_bias,Alex_Layer2_bias,Alex_Layer3_bias,Alex_Layer4_bias,
								Alex_Layer5_bias,Alex_Layer6_bias,Alex_Layer7_bias,Alex_Layer8_bias,
								Alex_Layer1_Weights,Alex_Layer2_Weights,Alex_Layer3_Weights,Alex_Layer4_Weights,
								Alex_Layer5_Weights, Alex_Layer6_Weights,Alex_Layer7_Weights,Alex_Layer8_Weights,
								Alex_Layer1_pool,Alex_Layer2_pool,Alex_Layer5_pool,
								Alex_Layer1_norm,Alex_Layer2_norm,Alex_Result_Neurons,
								Res_Layer1_Neurons,Res_Layer2_Neurons,Res_Layer3_Neurons,Res_Layer4_Neurons,
								Res_Layer5_Neurons,Res_Layer6_Neurons,Res_Layer7_Neurons,Res_Layer8_Neurons,
								Res_Layer9_Neurons,Res_Layer10_Neurons,Res_Layer11_Neurons,Res_Layer12_Neurons,
								Res_Layer13_Neurons,Res_Layer14_Neurons,Res_Layer15_Neurons,Res_Layer16_Neurons,
								Res_Layer17_Neurons,Res_Layer18_Neurons,
								Res_Layer1_Weights,Res_Layer2_Weights,Res_Layer3_Weights,Res_Layer4_Weights,
								Res_Layer5_Weights,Res_Layer6_Weights,Res_Layer7_Weights,Res_Layer8_Weights,
								Res_Layer9_Weights,Res_Layer10_Weights,Res_Layer11_Weights,Res_Layer12_Weights,
								Res_Layer13_Weights,Res_Layer14_Weights,Res_Layer15_Weights,Res_Layer16_Weights,
								Res_Layer17_Weights,Res_Block3_Weights,Res_Block4_Weights,Res_Block5_Weights,
								Res_Layer1_Gamma,Res_Layer2_Gamma,Res_Layer3_Gamma,Res_Layer4_Gamma,
								Res_Layer5_Gamma,Res_Layer6_Gamma,Res_Layer7_Gamma,Res_Layer8_Gamma,
								Res_Layer9_Gamma,Res_Layer10_Gamma,Res_Layer11_Gamma,Res_Layer12_Gamma,
								Res_Layer13_Gamma,Res_Layer14_Gamma,Res_Layer15_Gamma,Res_Layer16_Gamma,
								Res_Layer17_Gamma,Res_Block3_Gamma,Res_Block4_Gamma,Res_Block5_Gamma,
								Res_Layer1_Beta,Res_Layer2_Beta,Res_Layer3_Beta,Res_Layer4_Beta,
								Res_Layer5_Beta,Res_Layer6_Beta,Res_Layer7_Beta,Res_Layer8_Beta,
								Res_Layer9_Beta,Res_Layer10_Beta,Res_Layer11_Beta,Res_Layer12_Beta,
								Res_Layer13_Beta,Res_Layer14_Beta,Res_Layer15_Beta,Res_Layer16_Beta,
								Res_Layer17_Beta,Res_Block3_Beta,Res_Block4_Beta,Res_Block5_Beta,
								Res_mean1,Res_mean2,Res_mean3,Res_mean4,Res_mean5,
								Res_mean6,Res_mean7,Res_mean8,Res_mean9,Res_mean10,
								Res_mean11,Res_mean12,Res_mean13,Res_mean14,Res_mean15,
								Res_mean16,Res_mean17,Res_Block3_mean,Res_Block4_mean,Res_Block5_mean,
								Res_var1,Res_var2,Res_var3,Res_var4,Res_var5,
								Res_var6,Res_var7,Res_var8,Res_var9,Res_var10,
								Res_var11,Res_var12,Res_var13,Res_var14,Res_var15,
								Res_var16,Res_var17,Res_Block3_var,Res_Block4_var,Res_Block5_var,
								Res_FC_bias,Res_FC_Weights,
								Res_Layer3_basic,Res_Layer5_basic,Res_Layer7_basic,Res_Layer9_basic,
								Res_Layer11_basic,Res_Layer13_basic,Res_Layer15_basic,Res_Layer17_basic,
								Res_Block3_basic,Res_Block4_basic,Res_Block5_basic,
								Res_Layer1_bn,Res_Layer2_bn,Res_Layer3_bn,Res_Layer4_bn,
								Res_Layer5_bn,Res_Layer6_bn,Res_Layer7_bn,Res_Layer8_bn,
								Res_Layer9_bn,Res_Layer10_bn,Res_Layer11_bn,Res_Layer12_bn,
								Res_Layer13_bn,Res_Layer14_bn,Res_Layer15_bn,Res_Layer16_bn,
								Res_Layer17_bn,Res_Block3_bn,Res_Block4_bn,Res_Block5_bn,
								Res_Layer1_pool,Res_FC_Neurons,Res_Result_Neurons,
								2, 2,
								gridSize, i, b-i);
				}
			}

		}
	}
	
	Alex_Res_cudafree(Alex_Layer1_Neurons,Alex_Layer2_Neurons,Alex_Layer3_Neurons,Alex_Layer4_Neurons,
					Alex_Layer5_Neurons,Alex_Layer6_Neurons,Alex_Layer7_Neurons,Alex_Layer8_Neurons,
                    Alex_Layer1_bias,Alex_Layer2_bias,Alex_Layer3_bias,Alex_Layer4_bias,
                    Alex_Layer5_bias,Alex_Layer6_bias,Alex_Layer7_bias,Alex_Layer8_bias,
                    Alex_Layer1_Weights,Alex_Layer2_Weights,Alex_Layer3_Weights,Alex_Layer4_Weights,
                    Alex_Layer5_Weights, Alex_Layer6_Weights,Alex_Layer7_Weights,Alex_Layer8_Weights,
                    Alex_Layer1_pool,Alex_Layer2_pool,Alex_Layer5_pool,
					Alex_Layer1_norm,Alex_Layer2_norm,Alex_Result_Neurons,
					Res_Layer1_Neurons,Res_Layer2_Neurons,Res_Layer3_Neurons,Res_Layer4_Neurons,
					Res_Layer5_Neurons,Res_Layer6_Neurons,Res_Layer7_Neurons,Res_Layer8_Neurons,
					Res_Layer9_Neurons,Res_Layer10_Neurons,Res_Layer11_Neurons,Res_Layer12_Neurons,
					Res_Layer13_Neurons,Res_Layer14_Neurons,Res_Layer15_Neurons,Res_Layer16_Neurons,
					Res_Layer17_Neurons,Res_Layer18_Neurons,
                    Res_Layer1_Weights,Res_Layer2_Weights,Res_Layer3_Weights,Res_Layer4_Weights,
                    Res_Layer5_Weights,Res_Layer6_Weights,Res_Layer7_Weights,Res_Layer8_Weights,
                    Res_Layer9_Weights,Res_Layer10_Weights,Res_Layer11_Weights,Res_Layer12_Weights,
                    Res_Layer13_Weights,Res_Layer14_Weights,Res_Layer15_Weights,Res_Layer16_Weights,
                    Res_Layer17_Weights,Res_Block3_Weights,Res_Block4_Weights,Res_Block5_Weights,
                    Res_Layer1_Gamma,Res_Layer2_Gamma,Res_Layer3_Gamma,Res_Layer4_Gamma,
                    Res_Layer5_Gamma,Res_Layer6_Gamma,Res_Layer7_Gamma,Res_Layer8_Gamma,
                    Res_Layer9_Gamma,Res_Layer10_Gamma,Res_Layer11_Gamma,Res_Layer12_Gamma,
                    Res_Layer13_Gamma,Res_Layer14_Gamma,Res_Layer15_Gamma,Res_Layer16_Gamma,
                    Res_Layer17_Gamma,Res_Block3_Gamma,Res_Block4_Gamma,Res_Block5_Gamma,
                    Res_Layer1_Beta,Res_Layer2_Beta,Res_Layer3_Beta,Res_Layer4_Beta,
                    Res_Layer5_Beta,Res_Layer6_Beta,Res_Layer7_Beta,Res_Layer8_Beta,
                    Res_Layer9_Beta,Res_Layer10_Beta,Res_Layer11_Beta,Res_Layer12_Beta,
                    Res_Layer13_Beta,Res_Layer14_Beta,Res_Layer15_Beta,Res_Layer16_Beta,
                    Res_Layer17_Beta,Res_Block3_Beta,Res_Block4_Beta,Res_Block5_Beta,
                    Res_mean1,Res_mean2,Res_mean3,Res_mean4,Res_mean5,
                    Res_mean6,Res_mean7,Res_mean8,Res_mean9,Res_mean10,
                    Res_mean11,Res_mean12,Res_mean13,Res_mean14,Res_mean15,
                    Res_mean16,Res_mean17,Res_Block3_mean,Res_Block4_mean,Res_Block5_mean,
                    Res_var1,Res_var2,Res_var3,Res_var4,Res_var5,
                    Res_var6,Res_var7,Res_var8,Res_var9,Res_var10,
                    Res_var11,Res_var12,Res_var13,Res_var14,Res_var15,
                    Res_var16,Res_var17,Res_Block3_var,Res_Block4_var,Res_Block5_var,
                    Res_FC_bias,Res_FC_Weights,
					Res_Layer3_basic,Res_Layer5_basic,Res_Layer7_basic,Res_Layer9_basic,
					Res_Layer11_basic,Res_Layer13_basic,Res_Layer15_basic,Res_Layer17_basic,
					Res_Block3_basic,Res_Block4_basic,Res_Block5_basic,
					Res_Layer1_bn,Res_Layer2_bn,Res_Layer3_bn,Res_Layer4_bn,
					Res_Layer5_bn,Res_Layer6_bn,Res_Layer7_bn,Res_Layer8_bn,
					Res_Layer9_bn,Res_Layer10_bn,Res_Layer11_bn,Res_Layer12_bn,
					Res_Layer13_bn,Res_Layer14_bn,Res_Layer15_bn,Res_Layer16_bn,
					Res_Layer17_bn,Res_Block3_bn,Res_Block4_bn,Res_Block5_bn,
					Res_Layer1_pool,Res_FC_Neurons,Res_Result_Neurons);

	return 0;
}