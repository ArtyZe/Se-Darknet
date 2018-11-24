#include "route_layer.h"
#include "cuda.h"
#include "blas.h"
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "connected_layer.h"
#include "blas.h"
#include "gemm.h"
#include "activation_layer.h"

#include <stdio.h>

route_layer make_route_layer(int batch, int n, int *input_layers, int *input_sizes, int *out_chnel, int w, int h)
{
	fprintf(stderr,"route ");
    route_layer l = {0};
    l.type = ROUTE;
    l.batch = batch;
    l.n = n;
    l.input_layers = input_layers;
    l.input_sizes = input_sizes;
    int i;
    int outputs = 0;
    int out_c_insgesamt = 0;
    for(i = 0; i < n; ++i){
        fprintf(stderr," %d", input_layers[i]);
        outputs += input_sizes[i];
        out_c_insgesamt += out_chnel[i];
    }
    //outputs = input_sizes;

    l.outputs = outputs;
    l.inputs = outputs;
    l.delta =  calloc(outputs*batch, sizeof(float));
    l.delta_insert =  calloc(outputs*batch, sizeof(float));
    l.output = calloc(outputs*batch, sizeof(float));
    l.out_c = out_c_insgesamt;
    l.delta_channel = calloc(l.out_c, sizeof(float));
    l.out_w = w;
    l.out_h = h;
    l.channel_avg = calloc(out_c_insgesamt*batch, sizeof(float));
    l.channel_avg_insert = calloc(out_c_insgesamt*batch, sizeof(float));

    l.forward = forward_route_layer;
    l.backward = backward_route_layer;
    l.weights = calloc(l.out_c*l.out_c, sizeof(float));
	l.biases = calloc(l.out_c, sizeof(float));
    l.weight_updates = calloc(l.out_c*l.out_c, sizeof(float));
	l.bias_updates = calloc(l.out_c, sizeof(float));
	float scale = sqrt(2./l.out_c);
    //for(i = 0; i < l.out_c*l.out_c; ++i){
        //l.weights[i] = scale*rand_uniform(-1, 1);
    //}
		
    //for(i = 0; i < l.out_c; ++i){
        //l.biases[i] = 0;
    //}
    #ifdef GPU
    l.forward_gpu = forward_route_layer_gpu;
    l.backward_gpu = backward_route_layer_gpu;
    l.update_gpu = update_route_layer_gpu;

    l.weights_gpu = cuda_make_array(l.weights, l.out_c*l.out_c);
	l.biases_gpu = cuda_make_array(l.biases, l.out_c);
    l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.out_c*l.out_c);
	l.bias_updates_gpu = cuda_make_array(l.bias_updates, l.out_c);
    l.delta_gpu =  cuda_make_array(l.delta, outputs);
    l.delta_channel_gpu = cuda_make_array(l.delta_channel, l.out_c);
    l.delta_channel_insert_gpu = cuda_make_array(l.delta_insert, outputs*batch);
    l.output_gpu = cuda_make_array(l.output, outputs*batch);
    l.channel_avg_gpu = cuda_make_array(l.channel_avg, out_c_insgesamt*batch);
    l.channel_avg_insert_gpu = cuda_make_array(l.channel_avg_insert, out_c_insgesamt*batch);
    #endif
	fprintf(stderr, "\n");
    return l;
}

void resize_route_layer(route_layer *l, network *net)
{
    int i;
    layer first = net->layers[l->input_layers[0]];
    l->out_w = first.out_w;
    l->out_h = first.out_h;
    l->out_c = first.out_c;
    l->outputs = first.outputs;
    l->input_sizes[0] = first.outputs;
    for(i = 1; i < l->n; ++i){
        int index = l->input_layers[i];
        layer next = net->layers[index];
        l->outputs += next.outputs;
        l->input_sizes[i] = next.outputs;
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            l->out_c += next.out_c;
        }else{
            printf("%d %d, %d %d\n", next.out_w, next.out_h, first.out_w, first.out_h);
            l->out_h = l->out_w = l->out_c = 0;
        }
    }
    l->inputs = l->outputs;
    l->delta =  realloc(l->delta, l->outputs*l->batch*sizeof(float));
    l->output = realloc(l->output, l->outputs*l->batch*sizeof(float));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, l->outputs*l->batch);
    l->delta_gpu   = cuda_make_array(l->delta,  l->outputs*l->batch);
#endif
    
}

//get the avg_pooling value for every feature map
void get_route_parameter(layer l, float *channel_parameter)
{
	int j, k, chnl;
	float avg = 0.0;
	for(chnl = 0; chnl < l.c; ++chnl){
		float sum = 0.0;
		for(j = 0; j < l.h; ++j){
			for(k = 0; k < l.h; ++k){
				sum += l.output[chnl*l.w*l.h + j*l.w + k];
			}
		}
		avg = sum/(l.w*l.h);
		channel_parameter[chnl] = avg;
	}
}

//make normalisation 
void get_mean_parameter(const route_layer l)
{
	int i, j;
	float sum = 0.0;
	for(i = 0; i < l.out_c; i++)
	{
		sum += l.channel_avg[i];
	}
	for(j = 0; j < l.out_c; j++)
	{
		l.channel_avg[j] = l.channel_avg[j] / sum;
	}
}

//calculate the final output with weight
void get_final_output(const route_layer l)
{
	int i, j, k, c;
	for(i = 0; i < l.batch; ++i){
		for(c = 0; c < l.c; ++c){
			for(j = 0; j < l.h; ++j){
				for(k = 0; k < l.w; ++k){
					l.output[i*l.c*l.h*l.w + c*l.h*l.w + j*l.w + k] *= l.channel_avg[c];
				}
			}
		}
	}
}

void get_previos_delta(const route_layer l)
{
	int i, j, k, c;
	for(i = 0; i < l.batch; ++i){
		for(c = 0; c < l.c; ++c){
			for(j = 0; j < l.h; ++j){
				for(k = 0; k < l.w; ++k){
					l.delta[i*l.c*l.h*l.w + c*l.h*l.w + j*l.w + k] /= l.channel_avg[c];
				}
			}
		}
	}
}

void forward_route_layer(const route_layer l, network net)
{
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
    int i, j;
    int offset = 0;
    int offset_c = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *input = net.layers[index].output;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
			get_route_parameter(net.layers[index], l.channel_avg + offset_c + j*l.out_c);
            copy_cpu(input_size, input + j*input_size, 1, l.output + offset + j*l.outputs, 1);
        }
        offset += input_size;
        offset_c += net.layers[index].out_c;
    }
    get_mean_parameter(l);
    get_final_output(l);  
}

void backward_route_layer(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *delta = net.layers[index].delta;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
			get_previos_delta(l);
            axpy_cpu(input_size, 1, l.delta + offset + j*l.outputs, 1, delta + j*input_size, 1);
        }
        offset += input_size;
    }
}

void pull_route_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.out_c*l.out_c);
	cuda_pull_array(l.biases_gpu, l.biases, l.out_c);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.out_c*l.out_c);
	cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.out_c);
}

void push_route_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.out_c*l.out_c);
	cuda_push_array(l.biases_gpu, l.biases, l.out_c);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.out_c*l.out_c);
	cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.out_c);
}   


