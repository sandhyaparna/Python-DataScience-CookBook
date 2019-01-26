### check the time difference between cpu & gpu
import torch 
# cpu
t_cpu = torch.rand(500,500,500)
%timeit t_cpu @ t_cpu
# gpu
t_gpu = torch.rand(500,500,500).cuda()
%timeit t_gpu @ t_gpu










