#include <iostream>

#include "gms_fused_conv.cu"
#include "gms_fused_kernel.cu"
#include "gms_single_kernel.cu"

void Alex_Res_inference_thread(float *Alex_Layer1_Neurons,float *Alex_Layer2_Neurons,float *Alex_Layer3_Neurons,float *Alex_Layer4_Neurons,
					float *Alex_Layer5_Neurons,float *Alex_Layer6_Neurons,float *Alex_Layer7_Neurons,float *Alex_Layer8_Neurons,
                    float *Alex_Layer1_bias,float *Alex_Layer2_bias,float *Alex_Layer3_bias,float *Alex_Layer4_bias,
                    float *Alex_Layer5_bias,float *Alex_Layer6_bias,float *Alex_Layer7_bias,float *Alex_Layer8_bias,
                    float *Alex_Layer1_Weights,float *Alex_Layer2_Weights,float *Alex_Layer3_Weights,float *Alex_Layer4_Weights,
                    float *Alex_Layer5_Weights,float * Alex_Layer6_Weights,float *Alex_Layer7_Weights,float *Alex_Layer8_Weights,
                    float *Alex_Layer1_pool,float *Alex_Layer2_pool,float *Alex_Layer5_pool,
					float *Alex_Layer1_norm,float *Alex_Layer2_norm,float *Alex_Result_Neurons,
					float *Res_Layer1_Neurons,float *Res_Layer2_Neurons,float *Res_Layer3_Neurons,float *Res_Layer4_Neurons,
					float *Res_Layer5_Neurons,float *Res_Layer6_Neurons,float *Res_Layer7_Neurons,float *Res_Layer8_Neurons,
					float *Res_Layer9_Neurons,float *Res_Layer10_Neurons,float *Res_Layer11_Neurons,float *Res_Layer12_Neurons,
					float *Res_Layer13_Neurons,float *Res_Layer14_Neurons,float *Res_Layer15_Neurons,float *Res_Layer16_Neurons,
					float *Res_Layer17_Neurons,float *Res_Layer18_Neurons,
                    float *Res_Layer1_Weights,float *Res_Layer2_Weights,float *Res_Layer3_Weights,float *Res_Layer4_Weights,
                    float *Res_Layer5_Weights,float *Res_Layer6_Weights,float *Res_Layer7_Weights,float *Res_Layer8_Weights,
                    float *Res_Layer9_Weights,float *Res_Layer10_Weights,float *Res_Layer11_Weights,float *Res_Layer12_Weights,
                    float *Res_Layer13_Weights,float *Res_Layer14_Weights,float *Res_Layer15_Weights,float *Res_Layer16_Weights,
                    float *Res_Layer17_Weights,float *Res_Block3_Weights,float *Res_Block4_Weights,float *Res_Block5_Weights,
                    float *Res_Layer1_Gamma,float *Res_Layer2_Gamma,float *Res_Layer3_Gamma,float *Res_Layer4_Gamma,
                    float *Res_Layer5_Gamma,float *Res_Layer6_Gamma,float *Res_Layer7_Gamma,float *Res_Layer8_Gamma,
                    float *Res_Layer9_Gamma,float *Res_Layer10_Gamma,float *Res_Layer11_Gamma,float *Res_Layer12_Gamma,
                    float *Res_Layer13_Gamma,float *Res_Layer14_Gamma,float *Res_Layer15_Gamma,float *Res_Layer16_Gamma,
                    float *Res_Layer17_Gamma,float *Res_Block3_Gamma,float *Res_Block4_Gamma,float *Res_Block5_Gamma,
                    float *Res_Layer1_Beta,float *Res_Layer2_Beta,float *Res_Layer3_Beta,float *Res_Layer4_Beta,
                    float *Res_Layer5_Beta,float *Res_Layer6_Beta,float *Res_Layer7_Beta,float *Res_Layer8_Beta,
                    float *Res_Layer9_Beta,float *Res_Layer10_Beta,float *Res_Layer11_Beta,float *Res_Layer12_Beta,
                    float *Res_Layer13_Beta,float *Res_Layer14_Beta,float *Res_Layer15_Beta,float *Res_Layer16_Beta,
                    float *Res_Layer17_Beta,float *Res_Block3_Beta,float *Res_Block4_Beta,float *Res_Block5_Beta,
                    float *Res_mean1,float *Res_mean2,float *Res_mean3,float *Res_mean4,float *Res_mean5,
                    float *Res_mean6,float *Res_mean7,float *Res_mean8,float *Res_mean9,float *Res_mean10,
                    float *Res_mean11,float *Res_mean12,float *Res_mean13,float *Res_mean14,float *Res_mean15,
                    float *Res_mean16,float *Res_mean17,float *Res_Block3_mean,float *Res_Block4_mean,float *Res_Block5_mean,
                    float *Res_var1,float *Res_var2,float *Res_var3,float *Res_var4,float *Res_var5,
                    float *Res_var6,float *Res_var7,float *Res_var8,float *Res_var9,float *Res_var10,
                    float *Res_var11,float *Res_var12,float *Res_var13,float *Res_var14,float *Res_var15,
                    float *Res_var16,float *Res_var17,float *Res_Block3_var,float *Res_Block4_var,float *Res_Block5_var,
                    float *Res_FC_bias,float *Res_FC_Weights,
					float *Res_Layer3_basic,float *Res_Layer5_basic,float *Res_Layer7_basic,float *Res_Layer9_basic,
					float *Res_Layer11_basic,float *Res_Layer13_basic,float *Res_Layer15_basic,float *Res_Layer17_basic,
					float *Res_Block3_basic,float *Res_Block4_basic,float *Res_Block5_basic,
					float *Res_Layer1_bn,float *Res_Layer2_bn,float *Res_Layer3_bn,float *Res_Layer4_bn,
					float *Res_Layer5_bn,float *Res_Layer6_bn,float *Res_Layer7_bn,float *Res_Layer8_bn,
					float *Res_Layer9_bn,float *Res_Layer10_bn,float *Res_Layer11_bn,float *Res_Layer12_bn,
					float *Res_Layer13_bn,float *Res_Layer14_bn,float *Res_Layer15_bn,float *Res_Layer16_bn,
					float *Res_Layer17_bn,float *Res_Block3_bn,float *Res_Block4_bn,float *Res_Block5_bn,
					float *Res_Layer1_pool,float *Res_FC_Neurons,float *Res_Result_Neurons,
					int alex_num, int res_num,
					int gridSize, int alexWarpNum, int resWarpNum)
{
    // int gridSize = 0;
    int blockSize = max(alexWarpNum, resWarpNum);
    int alexNumOps = 0;
    int resNumOps = 0;

    int a_in_fm, r_in_fm, a_in_channel, r_in_channel;
    int a_out_fm, r_out_fm, a_out_channel, r_out_channel;
    int a_ker_fm, r_ker_fm;
    int a_str, r_str;
    int a_pad, r_pad;
    bool a_relu, r_relu;

    /* Fusing First convolution */
	// dim3 Block1(64,9,9);
	// dim3 Thread1(28,28,1);
	// fused_first_layer<<<Block1,Thread1>>>(Alex_Layer1_bias,Alex_Layer1_Weights,Res_Layer1_Weights,
	// 									Alex_Layer1_Neurons,Res_Layer1_Neurons,
	// 									Alex_Layer1_norm,Res_Layer1_bn,
	// 									alex_num,res_num,
	// 									224,55,4,2,11,3,
	// 									224,112,2,3,7,3,
	// 									5,11,
	// 									4,28);

    // gridSize = 264;
    dim3 Block1(gridSize,1,1);
    dim3 Thread1(32,blockSize,1);

    a_in_fm         = 224;  a_in_channel    = 3;
    a_out_fm        = 55;   a_out_channel   = 64;
    a_ker_fm        = 11;
    a_str           = 4;    a_pad           = 2;    a_relu          = true;

    r_in_fm         = 224;  r_in_channel    = 3;
    r_out_fm        = 112;  r_out_channel   = 64;
    r_ker_fm        = 7;    
    r_str           = 2;    r_pad           = 3;    r_relu          = true;

    alexNumOps = (a_out_fm*a_out_fm*a_out_channel - 1) / (gridSize * 32 * alexWarpNum) + 1;
    resNumOps = (r_out_fm*r_out_fm*r_out_channel - 1) / (gridSize * 32 * resWarpNum) + 1;

    // fused_first_conv_thread<<<Block1, Thread1>>>(Alex_Layer1_bias,Alex_Layer1_Weights,Res_Layer1_Weights,
    //                                 Alex_Layer1_Neurons,Res_Layer1_Neurons,
    //                                 Alex_Layer1_norm,Res_Layer1_bn,
    //                                 alex_num,res_num,
    //                                 a_in_fm,a_in_channel,a_out_fm,a_out_channel,a_ker_fm,a_str,a_pad,a_relu,
    //                                 r_in_fm,r_in_channel,r_out_fm,r_out_channel,r_ker_fm,r_str,r_pad,r_relu,
    //                                 alexWarpNum,resWarpNum,
    //                                 alexNumOps,resNumOps);

    /* Alex 1st lrm + Res 1st bn */
	dim3 Block2(64,8,8);
    dim3 Thread2(14,14,1);
    fused_lrm_bn1<<<Block2,Thread2>>>(Alex_Layer1_norm,Res_Layer1_bn,
									Alex_Layer1_pool,Res_Layer1_pool,
									alex_num,res_num,
									0.0001,0.75,5,55,
									Res_mean1,Res_var1,Res_Layer1_Gamma,Res_Layer1_Beta,112,true,
									64,5,11,
									64,8,14);

    /* Alex 1st max + Res 1st max */
    dim3 Block3(64,7,7);
    dim3 Thread3(9,9);
    fused_max1<<<Block3,Thread3>>>(Alex_Layer1_pool,Res_Layer1_pool,
                                    Alex_Layer2_Neurons,Res_Layer2_Neurons,
                                    alex_num,res_num,
                                    55,27,2,0,3,
                                    112,56,2,1,3,
                                    64,3,9,
                                    64,7,8);

    /* Alex 2nd conv + Res 2nd conv */
	/* Original */
	// dim3 Block4(192,8,8);
    // dim3 Thread4(9,9,1);
	// fused_two_conv<<<Block4,Thread4>>>(Alex_Layer2_bias,Alex_Layer2_Weights,Res_Layer2_Weights,
    //                                     Alex_Layer2_Neurons,Res_Layer2_Neurons,
    //                                     Alex_Layer2_norm,Res_Layer2_bn,
    //                                     alex_num,res_num,
    //                                     27,27,1,2,5,64,true,
    //                                     56,56,1,1,3,64,false,
    //                                     192,3,9,
    //                                     64,8,7);
	
    /*** test_conv_inner_thread ***/
    dim3 Block4(gridSize,1,1);
    dim3 Thread4(32,blockSize,1);

    a_in_fm         = 27;   a_in_channel    = 64;
    a_out_fm        = 27;   a_out_channel   = 192;
    a_ker_fm        = 5;
    a_str           = 1;    a_pad           = 2;    a_relu          = true;

    r_in_fm         = 56;   r_in_channel    = 64;
    r_out_fm        = 56;   r_out_channel   = 64;
    r_ker_fm        = 3;    
    r_str           = 1;    r_pad           = 1;    r_relu          = false;

    alexNumOps = (a_out_fm*a_out_fm*a_out_channel - 1) / (gridSize * 32 * alexWarpNum) + 1;
    resNumOps = (r_out_fm*r_out_fm*r_out_channel - 1) / (gridSize * 32 * resWarpNum) + 1;

    fused_two_conv_thread<<<Block4, Thread4>>>(Alex_Layer2_bias,Alex_Layer2_Weights,Res_Layer2_Weights,
                                    Alex_Layer2_Neurons,Res_Layer2_Neurons,
                                    Alex_Layer2_norm,Res_Layer2_bn,
                                    alex_num,res_num,
                                    a_in_fm,a_in_channel,a_out_fm,a_out_channel,a_ker_fm,a_str,a_pad,a_relu,
                                    r_in_fm,r_in_channel,r_out_fm,r_out_channel,r_ker_fm,r_str,r_pad,r_relu,
                                    alexWarpNum,resWarpNum,
                                    alexNumOps,resNumOps);


    /* Alex 2nd lrm + Res 2nd bn */
    dim3 Block5(192,2,2);
    dim3 Thread5(28,28);
    fused_lrm_bn1<<<Block5,Thread5>>>(Alex_Layer2_norm,Res_Layer2_bn,
                                    Alex_Layer2_pool,Res_Layer3_Neurons,
                                    alex_num,res_num,
                                    0.0001,0.75,5,27,
                                    Res_mean2,Res_var2,Res_Layer2_Gamma,Res_Layer2_Beta,56,true,
                                    192,1,27,
                                    64,2,28);


    /* Alex 2nd max */
    dim3 Block6(192,1,1);
    dim3 Thread6(13,13);
	max_jjb<<<Block6,Thread6>>>(Alex_Layer2_pool,Alex_Layer3_Neurons,alex_num,27,13,2,0,3);


    /* Alex 3rd conv + Res 3rd conv ********************************************************************************************************************/   
    // dim3 Block7(64,4,4);
    // dim3 Thread7(14,14,1);
	// fused_two_conv<<<Block7,Thread7>>>(Alex_Layer3_bias,Alex_Layer3_Weights,Res_Layer3_Weights,
    //                                     Alex_Layer3_Neurons,Res_Layer3_Neurons,
    //                                     Alex_Layer4_Neurons,Res_Layer3_bn,
    //                                     alex_num,res_num,
    //                                     13,13,1,1,3,192,true,
    //                                     56,56,1,1,3,64,false,
    //                                     64,1,13,
    //                                     64,4,14);

    /*** test_conv_inner_thread ***/
    dim3 Block7(gridSize,1,1);
    dim3 Thread7(32,blockSize,1);

    a_in_fm         = 13;   a_in_channel    = 192;
    a_out_fm        = 13;   a_out_channel   = 64;
    a_ker_fm        = 3;
    a_str           = 1;    a_pad           = 1;    a_relu          = true;

    r_in_fm         = 56;   r_in_channel    = 64;
    r_out_fm        = 56;   r_out_channel   = 64;
    r_ker_fm        = 3;    
    r_str           = 1;    r_pad           = 1;    r_relu          = false;
    
    alexNumOps = (a_out_fm*a_out_fm*a_out_channel - 1) / (gridSize * 32 * alexWarpNum) + 1;
    resNumOps = (r_out_fm*r_out_fm*r_out_channel - 1) / (gridSize * 32 * resWarpNum) + 1;

    fused_two_conv_thread<<<Block7, Thread7>>>(Alex_Layer3_bias,Alex_Layer3_Weights,Res_Layer3_Weights,
                                    Alex_Layer3_Neurons,Res_Layer3_Neurons,
                                    Alex_Layer4_Neurons,Res_Layer3_bn,
                                    alex_num,res_num,
                                    a_in_fm,a_in_channel,a_out_fm,a_out_channel,a_ker_fm,a_str,a_pad,a_relu,
                                    r_in_fm,r_in_channel,r_out_fm,r_out_channel,r_ker_fm,r_str,r_pad,r_relu,
                                    alexWarpNum,resWarpNum,
                                    alexNumOps,resNumOps);

	// dim3 Block7_1(320,1,1);
    // dim3 Thread7_1(13,13,1);
	// conv_jjb1<<<Block7_1,Thread7_1>>>(Alex_Layer3_bias,Alex_Layer3_Neurons,Alex_Layer3_Weights,Alex_Layer4_Neurons,alex_num,13,13,1,1,3,192,true,true);

    /* Res 3rd bn */
    dim3 Block8(64,8,8);
    dim3 Thread8(7,7);
 	batchnorm_jjb<<<Block8,Thread8>>>(Res_Layer3_bn,Res_Layer3_basic,res_num,Res_mean3,Res_var3,Res_Layer3_Gamma,Res_Layer3_Beta,56,false);
   
    /* Res 3rd basic */
    dim3 Block9(64,8,8);
    dim3 Thread9(7,7);
    basic_block_jjb<<<Block9,Thread9>>>(Res_Layer2_Neurons,Res_Layer3_basic,Res_Layer4_Neurons,res_num,56,true);


    /* Alex 4th conv + Res 4th conv */
    dim3 Block10(64,4,4);
    dim3 Thread10(14,14);
	fused_two_conv<<<Block10,Thread10>>>(Alex_Layer4_bias,Alex_Layer4_Weights,Res_Layer4_Weights,
                                            Alex_Layer4_Neurons,Res_Layer4_Neurons,
                                            Alex_Layer5_Neurons,Res_Layer4_bn,
                                            alex_num,res_num,
                                            13,13,1,1,3,384,true,
                                            56,56,1,1,3,64,false,
                                            64,1,13,
                                            64,4,14);
	// dim3 Block10_1(192,1,1);
    // dim3 Thread10_1(13,13,1);
	// conv_jjb1<<<Block10_1,Thread10_1>>>(Alex_Layer4_bias,Alex_Layer4_Neurons,Alex_Layer4_Weights,Alex_Layer5_Neurons,alex_num,13,13,1,1,3,384,true,true);

    /* Res 4th bn */
    dim3 Block11(64,7,7);
    dim3 Thread11(8,8);
	batchnorm_jjb<<<Block11,Thread11>>>(Res_Layer4_bn,Res_Layer5_Neurons,res_num,Res_mean4,Res_var4,Res_Layer4_Gamma,Res_Layer4_Beta,56,true);
	
    /* Alex 5th conv + Res 5th conv */
    dim3 Block12(64,4,4);
    dim3 Thread12(14,14);
	fused_two_conv<<<Block12,Thread12>>>(Alex_Layer5_bias,Alex_Layer5_Weights,Res_Layer5_Weights,
                                            Alex_Layer5_Neurons,Res_Layer5_Neurons,
                                            Alex_Layer5_pool,Res_Layer5_bn,
                                            alex_num,res_num,
                                            13,13,1,1,3,256,true,
                                            56,56,1,1,3,64,false,
                                            64,1,13,
                                            64,4,14);
	// dim3 Block12_1(192,1,1);
    // dim3 Thread12_1(13,13,1);									
	// conv_jjb1<<<Block12_1,Thread12_1>>>(Alex_Layer5_bias,Alex_Layer5_Neurons,Alex_Layer5_Weights,Alex_Layer5_pool,alex_num,13,13,1,1,3,256,true,true);
										
    /* Alex 5th max + Res 5th bn */
	dim3 Block13(256,7,7);
	dim3 Thread13(8,8);
	fused_bn_max1<<<Block13,Thread13>>>(Res_Layer5_bn,Alex_Layer5_pool,
	                                    Res_Layer5_basic,Alex_Layer6_Neurons,
	                                    res_num,alex_num,
	                                    Res_mean5,Res_var5,Res_Layer5_Gamma,Res_Layer5_Beta,56,false,
	                                    13,6,2,0,3,
										64,7,8,
	                                    256,1,6);


    /* Res 5th basic */
	dim3 Block14(64,8,8);
    dim3 Thread14(7,7);
    basic_block_jjb<<<Block14,Thread14>>>(Res_Layer4_Neurons,Res_Layer5_basic,Res_Layer6_Neurons,res_num,56,true);

   	//6th layer
	dim3 Block15(128,4,4);
    dim3 Thread15(7,7);
	conv_jjb<<<Block15,Thread15>>>(NULL,Res_Layer6_Neurons,Res_Layer6_Weights,Res_Layer6_bn,res_num,56,28,2,1,3,64,false,false);
	batchnorm_jjb<<<Block15,Thread15>>>(Res_Layer6_bn,Res_Layer7_Neurons,res_num,Res_mean6,Res_var6,Res_Layer6_Gamma,Res_Layer6_Beta,28,true);

	//7th layer
	conv_jjb<<<Block15,Thread15>>>(NULL,Res_Layer7_Neurons,Res_Layer7_Weights,Res_Layer7_bn,res_num,28,28,1,1,3,128,false,false);
	batchnorm_jjb<<<Block15,Thread15>>>(Res_Layer7_bn,Res_Layer7_basic,res_num,Res_mean7,Res_var7,Res_Layer7_Gamma,Res_Layer7_Beta,28,false);

	//Block B output
	conv_jjb<<<Block15,Thread15>>>(NULL,Res_Layer6_Neurons,Res_Block3_Weights,Res_Block3_bn,res_num,56,28,2,0,1,64,false,false); 
	batchnorm_jjb<<<Block15,Thread15>>>(Res_Block3_bn,Res_Block3_basic,res_num,Res_Block3_mean,Res_Block3_var,Res_Block3_Gamma,Res_Block3_Beta,28,false);

	basic_block_jjb<<<Block15,Thread15>>>(Res_Layer7_basic,Res_Block3_basic,Res_Layer8_Neurons,res_num,28,true);

	//8th layer
	conv_jjb<<<Block15,Thread15>>>(NULL,Res_Layer8_Neurons,Res_Layer8_Weights,Res_Layer8_bn,res_num,28,28,1,1,3,128,false,false);
	batchnorm_jjb<<<Block15,Thread15>>>(Res_Layer8_bn,Res_Layer9_Neurons,res_num,Res_mean8,Res_var8,Res_Layer8_Gamma,Res_Layer8_Beta,28,true);

	//9th layer
	conv_jjb<<<Block15,Thread15>>>(NULL,Res_Layer9_Neurons,Res_Layer9_Weights,Res_Layer9_bn,res_num,28,28,1,1,3,128,false,false);
	batchnorm_jjb<<<Block15,Thread15>>>(Res_Layer9_bn,Res_Layer9_basic,res_num,Res_mean9,Res_var9,Res_Layer9_Gamma,Res_Layer9_Beta,28,false);

	basic_block_jjb<<<Block15,Thread15>>>(Res_Layer8_Neurons,Res_Layer9_basic,Res_Layer10_Neurons,res_num,28,true);

    /* Res 10th conv */
    dim3 Block27(256,2,2);
    dim3 Thread27(7,7);
	conv_jjb<<<Block27,Thread27>>>(NULL,Res_Layer10_Neurons,Res_Layer10_Weights,Res_Layer10_bn,res_num,28,14,2,1,3,128,false,false);

    //10th layer
	conv_jjb<<<Block27,Thread27>>>(NULL,Res_Layer10_Neurons,Res_Layer10_Weights,Res_Layer10_bn,res_num,28,14,2,1,3,128,false,false);
	batchnorm_jjb<<<Block27,Thread27>>>(Res_Layer10_bn,Res_Layer11_Neurons,res_num,Res_mean10,Res_var10,Res_Layer10_Gamma,Res_Layer10_Beta,14,true);

	//11th layer
	conv_jjb<<<Block27,Thread27>>>(NULL,Res_Layer11_Neurons,Res_Layer11_Weights,Res_Layer11_bn,res_num,14,14,1,1,3,256,false,false);
	batchnorm_jjb<<<Block27,Thread27>>>(Res_Layer11_bn,Res_Layer11_basic,res_num,Res_mean11,Res_var11,Res_Layer11_Gamma,Res_Layer11_Beta,14,false);

    /* Res 11th bn */
	batchnorm_jjb<<<Block27,Thread27>>>(Res_Layer11_bn,Res_Layer11_basic,res_num,Res_mean11,Res_var11,Res_Layer11_Gamma,Res_Layer11_Beta,14,false);

    /* Res 11th block conv + bn + basic */
	conv_jjb<<<Block27,Thread27>>>(NULL,Res_Layer10_Neurons,Res_Block4_Weights,Res_Block4_bn,res_num,28,14,2,0,1,128,false,false);
	batchnorm_jjb<<<Block27,Thread27>>>(Res_Block4_bn,Res_Block4_basic,res_num,Res_Block4_mean,Res_Block4_var,Res_Block4_Gamma,Res_Block4_Beta,14,false);
	basic_block_jjb<<<Block27,Thread27>>>(Res_Layer11_basic,Res_Block4_basic,Res_Layer12_Neurons,res_num,14,true);

	//12th layer
	conv_jjb<<<Block27,Thread27>>>(NULL,Res_Layer12_Neurons,Res_Layer12_Weights,Res_Layer12_bn,res_num,14,14,1,1,3,256,false,false);
	batchnorm_jjb<<<Block27,Thread27>>>(Res_Layer12_bn,Res_Layer13_Neurons,res_num,Res_mean12,Res_var12,Res_Layer12_Gamma,Res_Layer12_Beta,14,true);

	//13th layer
	conv_jjb<<<Block27,Thread27>>>(NULL,Res_Layer13_Neurons,Res_Layer13_Weights,Res_Layer13_bn,res_num,14,14,1,1,3,256,false,false); 
	batchnorm_jjb<<<Block27,Thread27>>>(Res_Layer13_bn,Res_Layer13_basic,res_num,Res_mean13,Res_var13,Res_Layer13_Gamma,Res_Layer13_Beta,14,false);

	basic_block_jjb<<<Block27,Thread27>>>(Res_Layer12_Neurons,Res_Layer13_basic,Res_Layer14_Neurons,res_num,14,true);

    /* Res 14th ~ 17th + 18th avgpooling*/
    dim3 Block39(512,1,1);
    dim3 Thread39(7,7);
    // Res 14th 
	conv_jjb<<<Block39,Thread39>>>(NULL,Res_Layer14_Neurons,Res_Layer14_Weights,Res_Layer14_bn,res_num,14,7,2,1,3,256,false,false);
	batchnorm_jjb<<<Block39,Thread39>>>(Res_Layer14_bn,Res_Layer15_Neurons,res_num,Res_mean14,Res_var14,Res_Layer14_Gamma,Res_Layer14_Beta,7,true);

    // Res 15th
	conv_jjb<<<Block39,Thread39>>>(NULL,Res_Layer15_Neurons,Res_Layer15_Weights,Res_Layer15_bn,res_num,7,7,1,1,3,512,false,false);
	batchnorm_jjb<<<Block39,Thread39>>>(Res_Layer15_bn,Res_Layer15_basic,res_num,Res_mean15,Res_var15,Res_Layer15_Gamma,Res_Layer15_Beta,7,false);

	//Block D output
	conv_jjb<<<Block39,Thread39>>>(NULL,Res_Layer14_Neurons,Res_Block5_Weights,Res_Block5_bn,res_num,14,7,2,0,1,256,false,false);
	batchnorm_jjb<<<Block39,Thread39>>>(Res_Block5_bn,Res_Block5_basic,res_num,Res_Block5_mean,Res_Block5_var,Res_Block5_Gamma,Res_Block5_Beta,7,false);
	basic_block_jjb<<<Block39,Thread39>>>(Res_Layer15_basic,Res_Block5_basic,Res_Layer16_Neurons,res_num,7,true);

    // Res 16th
	conv_jjb<<<Block39,Thread39>>>(NULL,Res_Layer16_Neurons,Res_Layer16_Weights,Res_Layer16_bn,res_num,7,7,1,1,3,512,false,false);
	batchnorm_jjb<<<Block39,Thread39>>>(Res_Layer16_bn,Res_Layer17_Neurons,res_num,Res_mean16,Res_var16,Res_Layer16_Gamma,Res_Layer16_Beta,7,true);
	
    // Res 17th
	conv_jjb<<<Block39,Thread39>>>(NULL,Res_Layer17_Neurons,Res_Layer17_Weights,Res_Layer17_bn,res_num,7,7,1,1,3,512,false,false); 
	batchnorm_jjb<<<Block39,Thread39>>>(Res_Layer17_bn,Res_Layer17_basic,res_num,Res_mean17,Res_var17,Res_Layer17_Gamma,Res_Layer17_Beta,7,false);

	basic_block_jjb<<<Block39,Thread39>>>(Res_Layer16_Neurons,Res_Layer17_basic,Res_Layer18_Neurons,res_num,7,true);

    // Res 18th avgpooling
    dim3 Block40(512,1,1);
    dim3 Thread40(1,1);
	globalavg_jjb<<<Block40,Thread40>>>(Res_Layer18_Neurons,Res_FC_Neurons,res_num,7);

    /* Alex 6th fc */
    dim3 block41(4096,1,1);
    dim3 Thread41(1,1);

	fc_jjb<<<block41,Thread41>>>(Alex_Layer6_bias,Alex_Layer6_Neurons,Alex_Layer6_Weights,Alex_Layer7_Neurons,alex_num,(6*6*256),true);
    
	/* Alex 7th fc */
    dim3 block42(4096,1,1);
    dim3 Thread42(1,1);

	fc_jjb<<<block42,Thread42>>>(Alex_Layer7_bias,Alex_Layer7_Neurons,Alex_Layer7_Weights,Alex_Layer8_Neurons,alex_num,4096,true);

    /* Alex 8th fc + Res 18th fc */
    dim3 block43(1000,1,1);
    dim3 Thread43(1,1);
    fused_two_fc1<<<block43,Thread43>>>(Alex_Layer8_bias,Res_FC_bias,Alex_Layer8_Weights,Res_FC_Weights,
                                        Alex_Layer8_Neurons,Res_FC_Neurons,
                                        Alex_Result_Neurons,Res_Result_Neurons,
                                        alex_num,res_num,
                                        4096, false,
		                                512,false);



    for(int j = 0; j < alex_num; j++){
        float *Alex_Result_Neurons_CPU = (float *) malloc ((1000) * sizeof(float));
		cudaMemcpy(Alex_Result_Neurons_CPU, Alex_Result_Neurons, (1000) * sizeof(float), cudaMemcpyDeviceToHost);

		float max_alex = 0.0;
		int index_alex = 0;
		for(int i = 0; i < 1000; i++){
			if(max_alex < Alex_Result_Neurons_CPU[i]){
				max_alex = Alex_Result_Neurons_CPU[i];	
				index_alex = i;
			}
		}

		int line_count_alex = 0;
        char buffer_alex[1000];
        FILE *list_alex = fopen("imagenet1000_clsidx_to_labels.txt","rt");
        while(fgets(buffer_alex, 1000, list_alex) != NULL){
            line_count_alex++;
            if(line_count_alex == (index_alex+1)){
                printf("%f Alex: %s", max_alex, buffer_alex);
                // if (strcmp(buffer_alex, "Egyptian cat") != 0)
                // {
                //     printf("\n---Alexnet Result---");
                //     printf("\nClass ID: %d\nClass Name: %sProbability: %.20f\n\n", index_alex, buffer_alex, max_alex);
                //     exit(1);
                // }
                // printf("Alexnet: %d, %s", index_alex, buffer_alex);
                break;
            }
        }
        fclose(list_alex);
		// free(Alex_Result_Neurons_CPU);
    }


	for(int j = 0; j < res_num; j++){
        float *Res_Result_Neurons_CPU = (float *) malloc ((1000) * sizeof(float));
		cudaMemcpy(Res_Result_Neurons_CPU, Res_Result_Neurons, (1000) * sizeof(float), cudaMemcpyDeviceToHost);

		float max_res = 0.0;
        int index_res = 0; 
        for(int i = 0; i < 1000; i++){
            if(max_res < Res_Result_Neurons_CPU[i]){
                max_res = Res_Result_Neurons_CPU[i];	
                index_res = i;
            }
        }	
        int line_count_res = 0;
        char buffer_res[1000];
        FILE *list_res = fopen("imagenet1000_clsidx_to_labels.txt","rt");
        while(fgets(buffer_res, 1000, list_res) != NULL){
            line_count_res++;
            if(line_count_res == (index_res+1)){
                printf("%f Res: %s", max_res, buffer_res);
                // if (strcmp(buffer_res, "tabby, tabby cat") != 0)
                // {
                //     printf("\n---Resnet18 Result---");
                //     printf("\nClass ID: %d\nClass Name: %sProbability: %.20f\n\n", index_res, buffer_res, max_res);
                //     exit(1);
                // }
                // printf("Resnet18: %d, %s", index_res, buffer_res);
                break;
            }
        }
        fclose(list_res);
		// free(Res_Result_Neurons_CPU);
    }
}

void Alex_Res_inference_block(float *Alex_Layer1_Neurons,float *Alex_Layer2_Neurons,float *Alex_Layer3_Neurons,float *Alex_Layer4_Neurons,
					float *Alex_Layer5_Neurons,float *Alex_Layer6_Neurons,float *Alex_Layer7_Neurons,float *Alex_Layer8_Neurons,
                    float *Alex_Layer1_bias,float *Alex_Layer2_bias,float *Alex_Layer3_bias,float *Alex_Layer4_bias,
                    float *Alex_Layer5_bias,float *Alex_Layer6_bias,float *Alex_Layer7_bias,float *Alex_Layer8_bias,
                    float *Alex_Layer1_Weights,float *Alex_Layer2_Weights,float *Alex_Layer3_Weights,float *Alex_Layer4_Weights,
                    float *Alex_Layer5_Weights,float * Alex_Layer6_Weights,float *Alex_Layer7_Weights,float *Alex_Layer8_Weights,
                    float *Alex_Layer1_pool,float *Alex_Layer2_pool,float *Alex_Layer5_pool,
					float *Alex_Layer1_norm,float *Alex_Layer2_norm,float *Alex_Result_Neurons,
					float *Res_Layer1_Neurons,float *Res_Layer2_Neurons,float *Res_Layer3_Neurons,float *Res_Layer4_Neurons,
					float *Res_Layer5_Neurons,float *Res_Layer6_Neurons,float *Res_Layer7_Neurons,float *Res_Layer8_Neurons,
					float *Res_Layer9_Neurons,float *Res_Layer10_Neurons,float *Res_Layer11_Neurons,float *Res_Layer12_Neurons,
					float *Res_Layer13_Neurons,float *Res_Layer14_Neurons,float *Res_Layer15_Neurons,float *Res_Layer16_Neurons,
					float *Res_Layer17_Neurons,float *Res_Layer18_Neurons,
                    float *Res_Layer1_Weights,float *Res_Layer2_Weights,float *Res_Layer3_Weights,float *Res_Layer4_Weights,
                    float *Res_Layer5_Weights,float *Res_Layer6_Weights,float *Res_Layer7_Weights,float *Res_Layer8_Weights,
                    float *Res_Layer9_Weights,float *Res_Layer10_Weights,float *Res_Layer11_Weights,float *Res_Layer12_Weights,
                    float *Res_Layer13_Weights,float *Res_Layer14_Weights,float *Res_Layer15_Weights,float *Res_Layer16_Weights,
                    float *Res_Layer17_Weights,float *Res_Block3_Weights,float *Res_Block4_Weights,float *Res_Block5_Weights,
                    float *Res_Layer1_Gamma,float *Res_Layer2_Gamma,float *Res_Layer3_Gamma,float *Res_Layer4_Gamma,
                    float *Res_Layer5_Gamma,float *Res_Layer6_Gamma,float *Res_Layer7_Gamma,float *Res_Layer8_Gamma,
                    float *Res_Layer9_Gamma,float *Res_Layer10_Gamma,float *Res_Layer11_Gamma,float *Res_Layer12_Gamma,
                    float *Res_Layer13_Gamma,float *Res_Layer14_Gamma,float *Res_Layer15_Gamma,float *Res_Layer16_Gamma,
                    float *Res_Layer17_Gamma,float *Res_Block3_Gamma,float *Res_Block4_Gamma,float *Res_Block5_Gamma,
                    float *Res_Layer1_Beta,float *Res_Layer2_Beta,float *Res_Layer3_Beta,float *Res_Layer4_Beta,
                    float *Res_Layer5_Beta,float *Res_Layer6_Beta,float *Res_Layer7_Beta,float *Res_Layer8_Beta,
                    float *Res_Layer9_Beta,float *Res_Layer10_Beta,float *Res_Layer11_Beta,float *Res_Layer12_Beta,
                    float *Res_Layer13_Beta,float *Res_Layer14_Beta,float *Res_Layer15_Beta,float *Res_Layer16_Beta,
                    float *Res_Layer17_Beta,float *Res_Block3_Beta,float *Res_Block4_Beta,float *Res_Block5_Beta,
                    float *Res_mean1,float *Res_mean2,float *Res_mean3,float *Res_mean4,float *Res_mean5,
                    float *Res_mean6,float *Res_mean7,float *Res_mean8,float *Res_mean9,float *Res_mean10,
                    float *Res_mean11,float *Res_mean12,float *Res_mean13,float *Res_mean14,float *Res_mean15,
                    float *Res_mean16,float *Res_mean17,float *Res_Block3_mean,float *Res_Block4_mean,float *Res_Block5_mean,
                    float *Res_var1,float *Res_var2,float *Res_var3,float *Res_var4,float *Res_var5,
                    float *Res_var6,float *Res_var7,float *Res_var8,float *Res_var9,float *Res_var10,
                    float *Res_var11,float *Res_var12,float *Res_var13,float *Res_var14,float *Res_var15,
                    float *Res_var16,float *Res_var17,float *Res_Block3_var,float *Res_Block4_var,float *Res_Block5_var,
                    float *Res_FC_bias,float *Res_FC_Weights,
					float *Res_Layer3_basic,float *Res_Layer5_basic,float *Res_Layer7_basic,float *Res_Layer9_basic,
					float *Res_Layer11_basic,float *Res_Layer13_basic,float *Res_Layer15_basic,float *Res_Layer17_basic,
					float *Res_Block3_basic,float *Res_Block4_basic,float *Res_Block5_basic,
					float *Res_Layer1_bn,float *Res_Layer2_bn,float *Res_Layer3_bn,float *Res_Layer4_bn,
					float *Res_Layer5_bn,float *Res_Layer6_bn,float *Res_Layer7_bn,float *Res_Layer8_bn,
					float *Res_Layer9_bn,float *Res_Layer10_bn,float *Res_Layer11_bn,float *Res_Layer12_bn,
					float *Res_Layer13_bn,float *Res_Layer14_bn,float *Res_Layer15_bn,float *Res_Layer16_bn,
					float *Res_Layer17_bn,float *Res_Block3_bn,float *Res_Block4_bn,float *Res_Block5_bn,
					float *Res_Layer1_pool,float *Res_FC_Neurons,float *Res_Result_Neurons,
					int alex_num, int res_num,
                    int gridSize, int alexWarpNum, int resWarpNum) 
{
    // int gridSize = 0;
    int blockSize = alexWarpNum + resWarpNum;
    int alexNumOps = 0;
    int resNumOps = 0;

    int a_in_fm, r_in_fm, a_in_channel, r_in_channel;
    int a_out_fm, r_out_fm, a_out_channel, r_out_channel;
    int a_ker_fm, r_ker_fm;
    int a_str, r_str;
    int a_pad, r_pad;
    bool a_relu, r_relu;

    /*** Fusing First convolution ***/
    // gridSize = 264;
    dim3 Block1(gridSize,1,1);
    dim3 Thread1(32,blockSize,1);

    a_in_fm         = 224;  a_in_channel    = 3;
    a_out_fm        = 55;   a_out_channel   = 64;
    a_ker_fm        = 11;
    a_str           = 4;    a_pad           = 2;    a_relu          = true;

    r_in_fm         = 224;  r_in_channel    = 3;
    r_out_fm        = 112;  r_out_channel   = 64;
    r_ker_fm        = 7;    
    r_str           = 2;    r_pad           = 3;    r_relu          = true;

    alexNumOps = (a_out_fm*a_out_fm*a_out_channel - 1) / (gridSize * 32 * alexWarpNum) + 1;
    resNumOps = (r_out_fm*r_out_fm*r_out_channel - 1) / (gridSize * 32 * resWarpNum) + 1;

    fused_two_conv_block<<<Block1, Thread1>>>(Alex_Layer1_bias,Alex_Layer1_Weights,Res_Layer1_Weights,
                                    Alex_Layer1_Neurons,Res_Layer1_Neurons,
                                    Alex_Layer1_norm,Res_Layer1_bn,
                                    alex_num,res_num,
                                    a_in_fm,a_in_channel,a_out_fm,a_out_channel,a_ker_fm,a_str,a_pad,a_relu,
                                    r_in_fm,r_in_channel,r_out_fm,r_out_channel,r_ker_fm,r_str,r_pad,r_relu,
                                    alexWarpNum,resWarpNum,
                                    alexNumOps,resNumOps,
                                    true);

    // fused_two_conv_block_sh<<<Block1, Thread1>>>(Alex_Layer1_bias,Alex_Layer1_Weights,Res_Layer1_Weights,
    //                                 Alex_Layer1_Neurons,Res_Layer1_Neurons,
    //                                 Alex_Layer1_norm,Res_Layer1_bn,
    //                                 alex_num,res_num,
    //                                 a_in_fm,a_in_channel,a_out_fm,a_out_channel,a_ker_fm,a_str,a_pad,a_relu,
    //                                 r_in_fm,r_in_channel,r_out_fm,r_out_channel,r_ker_fm,r_str,r_pad,r_relu,
    //                                 alexWarpNum,resWarpNum,
    //                                 alexNumOps,resNumOps,
    //                                 100,0,
    //                                 true);

    // cudaError_t err{cudaGetLastError()};
    // std::cerr << cudaGetErrorString(err) << std::endl;

    /* Alex 1st lrm + Res 1st bn */
	dim3 Block2(64,8,8);
    dim3 Thread2(14,14,1);
    fused_lrm_bn1<<<Block2,Thread2>>>(Alex_Layer1_norm,Res_Layer1_bn,
									Alex_Layer1_pool,Res_Layer1_pool,
									alex_num,res_num,
									0.0001,0.75,5,55,
									Res_mean1,Res_var1,Res_Layer1_Gamma,Res_Layer1_Beta,112,true,
									64,5,11,
									64,8,14);

    /* Alex 1st max + Res 1st max */
    dim3 Block3(64,7,7);
    dim3 Thread3(9,9);
    fused_max1<<<Block3,Thread3>>>(Alex_Layer1_pool,Res_Layer1_pool,
                                    Alex_Layer2_Neurons,Res_Layer2_Neurons,
                                    alex_num,res_num,
                                    55,27,2,0,3,
                                    112,56,2,1,3,
                                    64,3,9,
                                    64,7,8);

    /************* Alex 2nd conv + Res 2nd conv *********************************************************************************************************/
	/*** Alex 2rd conv + Res 2rd conv ***/
    // gridSize = 4384;
    dim3 Block4(gridSize,1,1);
    dim3 Thread4(32,blockSize,1);

    a_in_fm         = 27;   a_in_channel    = 64;
    a_out_fm        = 27;   a_out_channel   = 192;
    a_ker_fm        = 5;
    a_str           = 1;    a_pad           = 2;    a_relu          = true;

    r_in_fm         = 56;   r_in_channel    = 64;
    r_out_fm        = 56;   r_out_channel   = 64;
    r_ker_fm        = 3;    
    r_str           = 1;    r_pad           = 1;    r_relu          = false;

    alexNumOps = (a_out_fm*a_out_fm*a_out_channel - 1) / (gridSize * 32 * alexWarpNum) + 1;
    resNumOps = (r_out_fm*r_out_fm*r_out_channel - 1) / (gridSize * 32 * resWarpNum) + 1;

    fused_two_conv_block<<<Block4, Thread4>>>(Alex_Layer2_bias,Alex_Layer2_Weights,Res_Layer2_Weights,
                                    Alex_Layer2_Neurons,Res_Layer2_Neurons,
                                    Alex_Layer2_norm,Res_Layer2_bn,
                                    alex_num,res_num,
                                    a_in_fm,a_in_channel,a_out_fm,a_out_channel,a_ker_fm,a_str,a_pad,a_relu,
                                    r_in_fm,r_in_channel,r_out_fm,r_out_channel,r_ker_fm,r_str,r_pad,r_relu,
                                    alexWarpNum,resWarpNum,
                                    alexNumOps,resNumOps,
                                    false);

    /* Alex 2nd lrm + Res 2nd bn */
    dim3 Block5(192,2,2);
    dim3 Thread5(28,28);
    fused_lrm_bn1<<<Block5,Thread5>>>(Alex_Layer2_norm,Res_Layer2_bn,
                                    Alex_Layer2_pool,Res_Layer3_Neurons,
                                    alex_num,res_num,
                                    0.0001,0.75,5,27,
                                    Res_mean2,Res_var2,Res_Layer2_Gamma,Res_Layer2_Beta,56,true,
                                    192,1,27,
                                    64,2,28);


    /* Alex 2nd max */
    dim3 Block6(192,1,1);
    dim3 Thread6(13,13);
	max_jjb<<<Block6,Thread6>>>(Alex_Layer2_pool,Alex_Layer3_Neurons,alex_num,27,13,2,0,3);


    /*** Alex 3rd conv + Res 3rd conv ***/
    // gridSize = 48;
    dim3 Block7(gridSize,1,1);
    dim3 Thread7(32,blockSize,1);

    a_in_fm         = 13;   a_in_channel    = 192;
    a_out_fm        = 13;   a_out_channel   = 384;
    a_ker_fm        = 3;
    a_str           = 1;    a_pad           = 1;    a_relu          = true;

    r_in_fm         = 56;   r_in_channel    = 64;
    r_out_fm        = 56;   r_out_channel   = 64;
    r_ker_fm        = 3;    
    r_str           = 1;    r_pad           = 1;    r_relu          = false;
    
    alexNumOps = (a_out_fm*a_out_fm*a_out_channel - 1) / (gridSize * 32 * alexWarpNum) + 1;
    resNumOps = (r_out_fm*r_out_fm*r_out_channel - 1) / (gridSize * 32 * resWarpNum) + 1;

    fused_two_conv_block<<<Block7, Thread7>>>(Alex_Layer3_bias,Alex_Layer3_Weights,Res_Layer3_Weights,
                                    Alex_Layer3_Neurons,Res_Layer3_Neurons,
                                    Alex_Layer4_Neurons,Res_Layer3_bn,
                                    alex_num,res_num,
                                    a_in_fm,a_in_channel,a_out_fm,a_out_channel,a_ker_fm,a_str,a_pad,a_relu,
                                    r_in_fm,r_in_channel,r_out_fm,r_out_channel,r_ker_fm,r_str,r_pad,r_relu,
                                    alexWarpNum,resWarpNum,
                                    alexNumOps,resNumOps,
                                    false);

    /* Res 3rd bn */
    dim3 Block8(64,8,8);
    dim3 Thread8(7,7);
 	batchnorm_jjb<<<Block8,Thread8>>>(Res_Layer3_bn,Res_Layer3_basic,res_num,Res_mean3,Res_var3,Res_Layer3_Gamma,Res_Layer3_Beta,56,false);
   
    /* Res 3rd basic */
    dim3 Block9(64,8,8);
    dim3 Thread9(7,7);
    basic_block_jjb<<<Block9,Thread9>>>(Res_Layer2_Neurons,Res_Layer3_basic,Res_Layer4_Neurons,res_num,56,true);


    /*** Alex 4th conv + Res 4th conv ***/
    // gridSize = 48;
    dim3 Block10(gridSize,1,1);
    dim3 Thread10(32,blockSize,1);

    a_in_fm         = 13;   a_in_channel    = 384;
    a_out_fm        = 13;   a_out_channel   = 256;
    a_ker_fm        = 3;
    a_str           = 1;    a_pad           = 1;    a_relu          = true;

    r_in_fm         = 56;   r_in_channel    = 64;
    r_out_fm        = 56;   r_out_channel   = 64;
    r_ker_fm        = 3;    
    r_str           = 1;    r_pad           = 1;    r_relu          = false;
    
    alexNumOps = (a_out_fm*a_out_fm*a_out_channel - 1) / (gridSize * 32 * alexWarpNum) + 1;
    resNumOps = (r_out_fm*r_out_fm*r_out_channel - 1) / (gridSize * 32 * resWarpNum) + 1;

    fused_two_conv_block<<<Block10, Thread10>>>(Alex_Layer4_bias,Alex_Layer4_Weights,Res_Layer4_Weights,
                                    Alex_Layer4_Neurons,Res_Layer4_Neurons,
                                    Alex_Layer5_Neurons,Res_Layer4_bn,
                                    alex_num,res_num,
                                    a_in_fm,a_in_channel,a_out_fm,a_out_channel,a_ker_fm,a_str,a_pad,a_relu,
                                    r_in_fm,r_in_channel,r_out_fm,r_out_channel,r_ker_fm,r_str,r_pad,r_relu,
                                    alexWarpNum,resWarpNum,
                                    alexNumOps,resNumOps,
                                    false);

    /* Res 4th bn */
    dim3 Block11(64,7,7);
    dim3 Thread11(8,8);
	batchnorm_jjb<<<Block11,Thread11>>>(Res_Layer4_bn,Res_Layer5_Neurons,res_num,Res_mean4,Res_var4,Res_Layer4_Gamma,Res_Layer4_Beta,56,true);
	

    /*** Alex 5th conv + Res 5th conv ***/
    // gridSize = 96;
    dim3 Block12(gridSize,1,1);
    dim3 Thread12(32,blockSize,1);

    a_in_fm         = 13;   a_in_channel    = 256;
    a_out_fm        = 13;   a_out_channel   = 256;
    a_ker_fm        = 3;
    a_str           = 1;    a_pad           = 1;    a_relu          = true;

    r_in_fm         = 56;   r_in_channel    = 64;
    r_out_fm        = 56;   r_out_channel   = 64;
    r_ker_fm        = 3;    
    r_str           = 1;    r_pad           = 1;    r_relu          = false;
    
    alexNumOps = (a_out_fm*a_out_fm*a_out_channel - 1) / (gridSize * 32 * alexWarpNum) + 1;
    resNumOps = (r_out_fm*r_out_fm*r_out_channel - 1) / (gridSize * 32 * resWarpNum) + 1;

    fused_two_conv_block<<<Block12, Thread12>>>(Alex_Layer5_bias,Alex_Layer5_Weights,Res_Layer5_Weights,
                                    Alex_Layer5_Neurons,Res_Layer5_Neurons,
                                    Alex_Layer5_pool,Res_Layer5_bn,
                                    alex_num,res_num,
                                    a_in_fm,a_in_channel,a_out_fm,a_out_channel,a_ker_fm,a_str,a_pad,a_relu,
                                    r_in_fm,r_in_channel,r_out_fm,r_out_channel,r_ker_fm,r_str,r_pad,r_relu,
                                    alexWarpNum,resWarpNum,
                                    alexNumOps,resNumOps,
                                    false);
					
    /* Alex 5th max + Res 5th bn */
	dim3 Block13(256,7,7);
	dim3 Thread13(8,8);
	fused_bn_max1<<<Block13,Thread13>>>(Res_Layer5_bn,Alex_Layer5_pool,
	                                    Res_Layer5_basic,Alex_Layer6_Neurons,
	                                    res_num,alex_num,
	                                    Res_mean5,Res_var5,Res_Layer5_Gamma,Res_Layer5_Beta,56,false,
	                                    13,6,2,0,3,
										64,7,8,
	                                    256,1,6);

    /* Res 5th basic */
	dim3 Block14(64,8,8);
    dim3 Thread14(7,7);
    basic_block_jjb<<<Block14,Thread14>>>(Res_Layer4_Neurons,Res_Layer5_basic,Res_Layer6_Neurons,res_num,56,true);

   	/* 6th ~ 10th layer */
    dim3 Block15(128, 4, 4);
    dim3 Thread15(7, 7, 1);

    gridSize = 48;
    blockSize = 32;
    dim3 Block_gms(gridSize, 1, 1);
    dim3 Thread_gms(32, blockSize, 1);
    int numOps = (28*28*128 - 1) / (gridSize * 32 * blockSize) + 1;

    //6th layer
    gms_conv<<<Block_gms,Thread_gms>>>(NULL,Res_Layer6_Neurons,Res_Layer6_Weights,Res_Layer6_bn,res_num,
                                56,64,
                                28,128,
                                3,2,1,false,false,numOps);
    // conv_jjb<<<Block15,Thread15>>>(NULL,Res_Layer6_Neurons,Res_Layer6_Weights,Res_Layer6_bn,res_num,56,28,2,1,3,64,false,false);
	batchnorm_jjb<<<Block15,Thread15>>>(Res_Layer6_bn,Res_Layer7_Neurons,res_num,Res_mean6,Res_var6,Res_Layer6_Gamma,Res_Layer6_Beta,28,true);

	//7th layer
	// conv_jjb<<<Block15,Thread15>>>(NULL,Res_Layer7_Neurons,Res_Layer7_Weights,Res_Layer7_bn,res_num,28,28,1,1,3,128,false,false);
    gms_conv<<<Block_gms,Thread_gms>>>(NULL,Res_Layer7_Neurons,Res_Layer7_Weights,Res_Layer7_bn,res_num,
                                28,128,
                                28,128,
                                3,1,1,false,false,numOps);
    batchnorm_jjb<<<Block15,Thread15>>>(Res_Layer7_bn,Res_Layer7_basic,res_num,Res_mean7,Res_var7,Res_Layer7_Gamma,Res_Layer7_Beta,28,false);

	//Block B output
	// conv_jjb<<<Block15,Thread15>>>(NULL,Res_Layer6_Neurons,Res_Block3_Weights,Res_Block3_bn,res_num,56,28,2,0,1,64,false,false); 
    gms_conv<<<Block_gms,Thread_gms>>>(NULL,Res_Layer6_Neurons,Res_Block3_Weights,Res_Block3_bn,res_num,
                                56,64,
                                28,128,
                                1,2,0,false,false,numOps);
	batchnorm_jjb<<<Block15,Thread15>>>(Res_Block3_bn,Res_Block3_basic,res_num,Res_Block3_mean,Res_Block3_var,Res_Block3_Gamma,Res_Block3_Beta,28,false);

	basic_block_jjb<<<Block15,Thread15>>>(Res_Layer7_basic,Res_Block3_basic,Res_Layer8_Neurons,res_num,28,true);

	//8th layer
	// conv_jjb<<<Block15,Thread15>>>(NULL,Res_Layer8_Neurons,Res_Layer8_Weights,Res_Layer8_bn,res_num,28,28,1,1,3,128,false,false);
    gms_conv<<<Block_gms,Thread_gms>>>(NULL,Res_Layer8_Neurons,Res_Layer8_Weights,Res_Layer8_bn,res_num,
                                28,128,
                                28,128,
                                3,1,1,false,false,numOps);
    batchnorm_jjb<<<Block15,Thread15>>>(Res_Layer8_bn,Res_Layer9_Neurons,res_num,Res_mean8,Res_var8,Res_Layer8_Gamma,Res_Layer8_Beta,28,true);

	//9th layer
	// conv_jjb<<<Block15,Thread15>>>(NULL,Res_Layer9_Neurons,Res_Layer9_Weights,Res_Layer9_bn,res_num,28,28,1,1,3,128,false,false);
    gms_conv<<<Block_gms,Thread_gms>>>(NULL,Res_Layer9_Neurons,Res_Layer9_Weights,Res_Layer9_bn,res_num,
                                28,128,
                                28,128,
                                3,1,1,false,false,numOps);
    batchnorm_jjb<<<Block15,Thread15>>>(Res_Layer9_bn,Res_Layer9_basic,res_num,Res_mean9,Res_var9,Res_Layer9_Gamma,Res_Layer9_Beta,28,false);

	basic_block_jjb<<<Block15,Thread15>>>(Res_Layer8_Neurons,Res_Layer9_basic,Res_Layer10_Neurons,res_num,28,true);

    /* 10th ~ 13th  */
    dim3 Block27(256,2,2);
    dim3 Thread27(7,7);

    gridSize = 48;
    blockSize = 16;
    dim3 Block_gms2(gridSize, 1, 1);
    dim3 Thread_gms2(32, blockSize, 1);
    numOps = (14*14*256 - 1) / (gridSize * 32 * blockSize) + 1;

    /* Res 10th conv */
	// conv_jjb<<<Block27,Thread27>>>(NULL,Res_Layer10_Neurons,Res_Layer10_Weights,Res_Layer10_bn,res_num,28,14,2,1,3,128,false,false);
    gms_conv<<<Block_gms2,Thread_gms2>>>(NULL,Res_Layer10_Neurons,Res_Layer10_Weights,Res_Layer10_bn,res_num,
                                28,128,
                                14,256,
                                3,2,1,false,false,numOps);
	batchnorm_jjb<<<Block27,Thread27>>>(Res_Layer10_bn,Res_Layer11_Neurons,res_num,Res_mean10,Res_var10,Res_Layer10_Gamma,Res_Layer10_Beta,14,true);

	//11th layer
	// conv_jjb<<<Block27,Thread27>>>(NULL,Res_Layer11_Neurons,Res_Layer11_Weights,Res_Layer11_bn,res_num,14,14,1,1,3,256,false,false);
    gms_conv<<<Block_gms2,Thread_gms2>>>(NULL,Res_Layer11_Neurons,Res_Layer11_Weights,Res_Layer11_bn,res_num,
                                14,256,
                                14,256,
                                3,1,1,false,false,numOps);
    batchnorm_jjb<<<Block27,Thread27>>>(Res_Layer11_bn,Res_Layer11_basic,res_num,Res_mean11,Res_var11,Res_Layer11_Gamma,Res_Layer11_Beta,14,false);

    /* Res 11th bn */
	batchnorm_jjb<<<Block27,Thread27>>>(Res_Layer11_bn,Res_Layer11_basic,res_num,Res_mean11,Res_var11,Res_Layer11_Gamma,Res_Layer11_Beta,14,false);

    /* Res 11th block conv + bn + basic */
	// conv_jjb<<<Block27,Thread27>>>(NULL,Res_Layer10_Neurons,Res_Block4_Weights,Res_Block4_bn,res_num,28,14,2,0,1,128,false,false);
    gms_conv<<<Block_gms2,Thread_gms2>>>(NULL,Res_Layer10_Neurons,Res_Block4_Weights,Res_Block4_bn,res_num,
                                28,128,
                                14,256,
                                1,2,0,false,false,numOps);
    batchnorm_jjb<<<Block27,Thread27>>>(Res_Block4_bn,Res_Block4_basic,res_num,Res_Block4_mean,Res_Block4_var,Res_Block4_Gamma,Res_Block4_Beta,14,false);
	basic_block_jjb<<<Block27,Thread27>>>(Res_Layer11_basic,Res_Block4_basic,Res_Layer12_Neurons,res_num,14,true);

	//12th layer
	// conv_jjb<<<Block27,Thread27>>>(NULL,Res_Layer12_Neurons,Res_Layer12_Weights,Res_Layer12_bn,res_num,14,14,1,1,3,256,false,false);
    gms_conv<<<Block_gms2,Thread_gms2>>>(NULL,Res_Layer12_Neurons,Res_Layer12_Weights,Res_Layer12_bn,res_num,
                                14,256,
                                14,256,
                                3,1,1,false,false,numOps);
    batchnorm_jjb<<<Block27,Thread27>>>(Res_Layer12_bn,Res_Layer13_Neurons,res_num,Res_mean12,Res_var12,Res_Layer12_Gamma,Res_Layer12_Beta,14,true);

	//13th layer
	// conv_jjb<<<Block27,Thread27>>>(NULL,Res_Layer13_Neurons,Res_Layer13_Weights,Res_Layer13_bn,res_num,14,14,1,1,3,256,false,false); 
    gms_conv<<<Block_gms2,Thread_gms2>>>(NULL,Res_Layer13_Neurons,Res_Layer13_Weights,Res_Layer13_bn,res_num,
                                14,256,
                                14,256,
                                3,1,1,false,false,numOps);
    batchnorm_jjb<<<Block27,Thread27>>>(Res_Layer13_bn,Res_Layer13_basic,res_num,Res_mean13,Res_var13,Res_Layer13_Gamma,Res_Layer13_Beta,14,false);

	basic_block_jjb<<<Block27,Thread27>>>(Res_Layer12_Neurons,Res_Layer13_basic,Res_Layer14_Neurons,res_num,14,true);

    /* Res 14th ~ 17th + 18th avgpooling*/
    dim3 Block39(512,1,1);
    dim3 Thread39(7,7);

    numOps = (7*7*512 - 1) / (gridSize * 32 * blockSize) + 1;

    // Res 14th 
	// conv_jjb<<<Block39,Thread39>>>(NULL,Res_Layer14_Neurons,Res_Layer14_Weights,Res_Layer14_bn,res_num,14,7,2,1,3,256,false,false);
    gms_conv<<<Block_gms,Thread_gms>>>(NULL,Res_Layer14_Neurons,Res_Layer14_Weights,Res_Layer14_bn,res_num,
                                14,256,
                                7,512,
                                3,2,1,false,false,numOps);
    batchnorm_jjb<<<Block39,Thread39>>>(Res_Layer14_bn,Res_Layer15_Neurons,res_num,Res_mean14,Res_var14,Res_Layer14_Gamma,Res_Layer14_Beta,7,true);

    // Res 15th
	// conv_jjb<<<Block39,Thread39>>>(NULL,Res_Layer15_Neurons,Res_Layer15_Weights,Res_Layer15_bn,res_num,7,7,1,1,3,512,false,false);
    gms_conv<<<Block_gms,Thread_gms>>>(NULL,Res_Layer15_Neurons,Res_Layer15_Weights,Res_Layer15_bn,res_num,
                                7,512,
                                7,512,
                                3,1,1,false,false,numOps);
    batchnorm_jjb<<<Block39,Thread39>>>(Res_Layer15_bn,Res_Layer15_basic,res_num,Res_mean15,Res_var15,Res_Layer15_Gamma,Res_Layer15_Beta,7,false);

	//Block D output
	// conv_jjb<<<Block39,Thread39>>>(NULL,Res_Layer14_Neurons,Res_Block5_Weights,Res_Block5_bn,res_num,14,7,2,0,1,256,false,false);
    gms_conv<<<Block_gms,Thread_gms>>>(NULL,Res_Layer14_Neurons,Res_Block5_Weights,Res_Block5_bn,res_num,
                                14,256,
                                7,512,
                                1,2,0,false,false,numOps);
    batchnorm_jjb<<<Block39,Thread39>>>(Res_Block5_bn,Res_Block5_basic,res_num,Res_Block5_mean,Res_Block5_var,Res_Block5_Gamma,Res_Block5_Beta,7,false);
	basic_block_jjb<<<Block39,Thread39>>>(Res_Layer15_basic,Res_Block5_basic,Res_Layer16_Neurons,res_num,7,true);

    // Res 16th
	// conv_jjb<<<Block39,Thread39>>>(NULL,Res_Layer16_Neurons,Res_Layer16_Weights,Res_Layer16_bn,res_num,7,7,1,1,3,512,false,false);
    gms_conv<<<Block_gms,Thread_gms>>>(NULL,Res_Layer16_Neurons,Res_Layer16_Weights,Res_Layer16_bn,res_num,
                                7,512,
                                7,512,
                                3,1,1,false,false,numOps);
    batchnorm_jjb<<<Block39,Thread39>>>(Res_Layer16_bn,Res_Layer17_Neurons,res_num,Res_mean16,Res_var16,Res_Layer16_Gamma,Res_Layer16_Beta,7,true);
	
    // Res 17th
	// conv_jjb<<<Block39,Thread39>>>(NULL,Res_Layer17_Neurons,Res_Layer17_Weights,Res_Layer17_bn,res_num,7,7,1,1,3,512,false,false); 
    gms_conv<<<Block_gms,Thread_gms>>>(NULL,Res_Layer17_Neurons,Res_Layer17_Weights,Res_Layer17_bn,res_num,
                                7,512,
                                7,512,
                                3,1,1,false,false,numOps);
    batchnorm_jjb<<<Block39,Thread39>>>(Res_Layer17_bn,Res_Layer17_basic,res_num,Res_mean17,Res_var17,Res_Layer17_Gamma,Res_Layer17_Beta,7,false);

	basic_block_jjb<<<Block39,Thread39>>>(Res_Layer16_Neurons,Res_Layer17_basic,Res_Layer18_Neurons,res_num,7,true);

    // Res 18th avgpooling
    dim3 Block40(512,1,1);
    dim3 Thread40(1,1);
	globalavg_jjb<<<Block40,Thread40>>>(Res_Layer18_Neurons,Res_FC_Neurons,res_num,7);

    /* Alex 6th fc */
    dim3 block41(4096,1,1);
    dim3 Thread41(1,1);

	fc_jjb<<<block41,Thread41>>>(Alex_Layer6_bias,Alex_Layer6_Neurons,Alex_Layer6_Weights,Alex_Layer7_Neurons,alex_num,(6*6*256),true);
    
	/* Alex 7th fc */
    dim3 block42(4096,1,1);
    dim3 Thread42(1,1);

	fc_jjb<<<block42,Thread42>>>(Alex_Layer7_bias,Alex_Layer7_Neurons,Alex_Layer7_Weights,Alex_Layer8_Neurons,alex_num,4096,true);

	

    /* Alex 8th fc + Res 18th fc */
    dim3 block43(1000,1,1);
    dim3 Thread43(1,1);
    fused_two_fc1<<<block43,Thread43>>>(Alex_Layer8_bias,Res_FC_bias,Alex_Layer8_Weights,Res_FC_Weights,
                                        Alex_Layer8_Neurons,Res_FC_Neurons,
                                        Alex_Result_Neurons,Res_Result_Neurons,
                                        alex_num,res_num,
                                        4096, false,
		                                512,false);



    for(int j = 0; j < alex_num; j++){
        float *Alex_Result_Neurons_CPU = (float *) malloc ((1000) * sizeof(float));
		cudaMemcpy(Alex_Result_Neurons_CPU, Alex_Result_Neurons, (1000) * sizeof(float), cudaMemcpyDeviceToHost);

		float max_alex = 0.0;
		int index_alex = 0;
		for(int i = 0; i < 1000; i++){
			if(max_alex < Alex_Result_Neurons_CPU[i]){
				max_alex = Alex_Result_Neurons_CPU[i];	
				index_alex = i;
			}
		}

		int line_count_alex = 0;
        char buffer_alex[1000];
        FILE *list_alex = fopen("imagenet1000_clsidx_to_labels.txt","rt");
        while(fgets(buffer_alex, 1000, list_alex) != NULL){
            line_count_alex++;
            if(line_count_alex == (index_alex+1)){
                printf("%f Alex: %s", max_alex, buffer_alex);
                // if (max_alex != 17.64119338989257812500F)
                // {
                //     printf("\n---Alexnet Result---");
                //     printf("\nClass ID: %d\nClass Name: %sProbability: %.20f\n\n", index_alex, buffer_alex, max_alex);
                //     exit(1);
                // }
                // printf("Alexnet: %d, %s", index_alex, buffer_alex);
                break;
            }
        }
        fclose(list_alex);
		// free(Alex_Result_Neurons_CPU);
    }


	for(int j = 0; j < res_num; j++){
        float *Res_Result_Neurons_CPU = (float *) malloc ((1000) * sizeof(float));
		cudaMemcpy(Res_Result_Neurons_CPU, Res_Result_Neurons, (1000) * sizeof(float), cudaMemcpyDeviceToHost);

		float max_res = 0.0;
        int index_res = 0; 
        for(int i = 0; i < 1000; i++){
            if(max_res < Res_Result_Neurons_CPU[i]){
                max_res = Res_Result_Neurons_CPU[i];	
                index_res = i;
            }
        }	
        int line_count_res = 0;
        char buffer_res[1000];
        FILE *list_res = fopen("imagenet1000_clsidx_to_labels.txt","rt");
        while(fgets(buffer_res, 1000, list_res) != NULL){
            line_count_res++;
            if(line_count_res == (index_res+1)){
                printf("%f Res: %s", max_res, buffer_res);
                // if (max_res != 10.29121589660644531250F)
                // {
                //     printf("\n---Resnet18 Result---");
                //     printf("\nClass ID: %d\nClass Name: %sProbability: %.20f\n\n", index_res, buffer_res, max_res);
                //     exit(1);
                // }
                // printf("Resnet18: %d, %s", index_res, buffer_res);
                break;
            }
        }
        fclose(list_res);
		// free(Res_Result_Neurons_CPU);
    }
}