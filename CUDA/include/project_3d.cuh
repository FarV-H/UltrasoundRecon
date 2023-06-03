__global__ void Mul_weight_data_kernel(float* weight, float* data, float* result, unsigned long data_size, unsigned long block_size) {
    unsigned long Dataidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long Projidx = blockIdx.y * blockDim.y + threadIdx.y;

	if (Dataidx >= data_size || Projidx >= block_size)
	{
		return;
	}
	result[Projidx * data_size + Dataidx] = weight[Projidx * data_size + Dataidx] * data[Dataidx];
}