# import onnx
# import onnx_tensorrt.backend as backend

import torch
import torch.nn as nn
import numpy as np
points_num = 10000
# model = onnx.load("./onnx-file/model_sim.onnx")
# onnx.checker.check_model(model)
# engine = backend.prepare(model, device='CUDA:0')
input_data = torch.randn(size=(2, 15, points_num), device='cpu').numpy()
a_s = torch.randn(size=(2, points_num, points_num), device='cpu').numpy()
a_l = torch.randn(size=(2, points_num, points_num), device='cpu').numpy()
output = np.empty((2, points_num, 17), dtype=np.float16)

# tmp = [input_data, a_s, a_l]

# output_data = engine.run(inputs=[input_data, a_s, a_l])[0]
# print(output_data)
# print(output_data.shape)

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

import common

onnx_file_path = "./onnx-file/model_sim.onnx"

TRT_LOGGER = trt.Logger()

builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(common.EXPLICIT_BATCH)
parser = trt.OnnxParser(network, TRT_LOGGER)
runtime = trt.Runtime(TRT_LOGGER)

# Parse model file
with open(onnx_file_path, "rb") as model:
    print("Beginning ONNX file parsing")
    if not parser.parse(model.read()):
        print("ERROR: Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
print("Completed parsing of ONNX file")

opt_profile = builder.create_optimization_profile()
opt_profile.set_shape(input='input', min=(1,15,1000), opt=(1,15,5000), max=(1,15,10000))
opt_profile.set_shape(input='a_s', min=(1,1000,1000), opt=(1,5000,5000), max=(1,10000,10000))
opt_profile.set_shape(input='a_l', min=(1,1000,1000), opt=(1,5000,5000), max=(1,10000,10000))

config = builder.create_builder_config()

config.add_optimization_profile(opt_profile)

config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28) # 256MiB*2*2*2*2

plan = builder.build_serialized_network(network, config)

engine = runtime.deserialize_cuda_engine(plan)

context = engine.create_execution_context()

print("aaaaaaaaa")

inputs, outputs, bindings, stream = common.allocate_buffers(engine)
print(inputs)
# print(engine.get_binding_shape(0))
# print(engine.get_binding_shape(1))
# print(engine.get_binding_shape(2))
# print(engine.get_binding_shape(3))

# d_input = cuda.mem_alloc(1 * input_data.size * input_data.dtype.itemsize)
# d_a_s = cuda.mem_alloc(1 * a_s.size * a_s.dtype.itemsize)
# d_a_l = cuda.mem_alloc(1 * a_l.size * a_l.dtype.itemsize)
# d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)

# bindings = [int(d_input), int(d_a_s), int(d_a_l), int(d_output)]

# stream = cuda.Stream()

# cuda.memcpy_htod_async(d_input, input_data, stream)
# cuda.memcpy_htod_async(d_a_s, a_s, stream)
# cuda.memcpy_htod_async(d_a_l, a_l, stream)
# context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
# # 从GPU中返回output。
# cuda.memcpy_dtoh_async(output, d_output, stream)
# # 同步流。
# stream.synchronize()
# print(output)


