#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "route_layer.h"
#include "cuda.h"
}



__global__ void backward_route_layer_kernel_step2(int n, int w, int h, int c, float *in_delta, float *channel_avg)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c*b);
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        in_delta[in_index] /= channel_avg[out_index];
    }
}

__global__ void backward_route_layer_kernel_step1(int n, int w, int h, int c, float *in_delta, float *out_delta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;
    
    int k = 0;
    int p = id% c;
    id /= c;
    int b = id;
    
	int out_index = (p + c*b);
	//calculate delta in weights way
	for(k = 0; k < w*h; ++k){
		int in_index = k + h*w*(p + b*c);
		out_delta[out_index] += in_delta[in_index];
	}
}

__global__ void forward_route_layer_kernel_step1(int n, int w, int h, int c, float *input, float *channel_avg, float* channel_avg_insert, float* weights)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

	//if(id != 0)
	//{
		//printf("the current thread is %d and max is %d\n", id, n);
	//}
    int k = id % c;  //channel
    id /= c;
    int b = id;    //batch
    
	//printf("batch number is %d, channel number is %d\n", b, k);
    int i;
    int out_index = (k + c*b);
    channel_avg[out_index] = 0;
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        channel_avg[out_index] += input[in_index];
        //printf("the output value is %f\n", input[in_index]);
    }
    channel_avg[out_index] /= w*h;

}

__global__ void forward_route_layer_kernel_step2(int n, int w, int h, int c, float *output, float *channel_avg, float* channel_avg_insert, float* weights)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;
    
    int k = id % c;  //channel
    id /= c;
    int b = id;    //batch
    
	//printf("batch number is %d, channel number is %d\n", b, k);
    int i;
    int channel_index = (k + c*b);
    for(i = 0; i < w*h; ++i){
        int out_index = i + h*w*(k + b*c);
        output[out_index] *= channel_avg_insert[channel_index];
        //printf("the output value is %f\n", input[in_index]);
    }
}

extern "C" void forward_route_layer_gpu(route_layer l, network net)
{
    int i, j;
    int offset = 0;
   // printf("start to forward route layer\n");
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *input = net.layers[index].output_gpu;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            copy_gpu(input_size, input + j*input_size, 1, l.output_gpu + offset + j*l.outputs, 1);
        }
        offset += input_size;
    }
    
    if(l.n > 1)
	{
		size_t n = l.out_c*l.batch;
		forward_route_layer_kernel_step1<<<cuda_gridsize(n), BLOCK>>>(n, l.out_w, l.out_h, l.out_c, l.output_gpu, l.channel_avg_gpu, l.channel_avg_insert_gpu, l.weights_gpu);
		check_error(cudaPeekAtLastError());
		
		//connection layer
		gemm_gpu(0, 1, 1, l.out_c, l.out_c, 1, l.channel_avg_gpu, l.out_c, l.weights_gpu, l.out_c, 1, l.channel_avg_insert_gpu, l.out_c);
    
		//selu layer
		activate_array_gpu(l.channel_avg_insert_gpu, l.out_c, SELU);
    
		//connection layer
		//gemm_gpu(0, 1, 1, c, c/16, 1, channel_avg_insert, c/16, weights, c, 1, channel_avg_insert, c/16);
		//printf("the output channel is %d\n", l.out_c);
		forward_route_layer_kernel_step2<<<cuda_gridsize(n), BLOCK>>>(n, l.out_w, l.out_h, l.out_c, l.output_gpu, l.channel_avg_gpu, l.channel_avg_insert_gpu, l.weights_gpu);		
		
		//printf("finished to forward route layer\n");	
	}
}

extern "C" void backward_route_layer_gpu(route_layer l, network net)
{
    //printf("start to backward route layer\n");
    if(l.n > 1)
    {
		//here is not l.delta_gpu, should change it
		constrain_gpu(l.out_w*l.out_h*l.batch, 1, l.delta_gpu, 1);
		size_t n = l.out_c*l.batch;
		backward_route_layer_kernel_step1<<<cuda_gridsize(n), BLOCK>>>(n, l.out_w, l.out_h, l.out_c, l.delta_gpu, l.delta_channel_gpu);
		//activation backward
		gradient_array_gpu(l.channel_avg_insert_gpu, l.out_c*l.batch, SELU, l.delta_channel_gpu);
		//gemm backward
		gemm_gpu(1, 0, l.out_c, l.out_c, 1, 1, l.delta_channel_gpu, l.out_c, l.channel_avg_gpu, l.out_c, 1,l.weight_updates_gpu, l.out_c);
		
		backward_route_layer_kernel_step2<<<cuda_gridsize(n), BLOCK>>>(n, l.out_w, l.out_h, l.out_c, l.delta_gpu, l.channel_avg_gpu);
		check_error(cudaPeekAtLastError());
	}
    
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *delta = net.layers[index].delta_gpu;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            axpy_gpu(input_size, 1, l.delta_gpu + offset + j*l.outputs, 1, delta + j*input_size, 1);
        }
        offset += input_size;
    }
    //printf("finished to backward route layer\n");
    //cuda_pull_array(l.delta_gpu, net.delta, l.batch*l.inputs);
    //backward_route_layer(l, net);
    //cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    //printf("hier finished to backward route layer\n");
}

extern "C" void update_route_layer_gpu(layer l, update_args a)
{
	float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.inputs*l.outputs, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
        }
    }else{

        //if(l.batch_normalize){
            //axpy_gpu(l.outputs, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            //scal_gpu(l.outputs, momentum, l.scale_updates_gpu, 1);
        //}

        axpy_gpu(l.out_c*l.out_c, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(l.out_c*l.out_c, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(l.out_c*l.out_c, momentum, l.weight_updates_gpu, 1);
    }
} 


    
    
    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
