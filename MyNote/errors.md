```python
RuntimeError: CUDA out of memory. Tried to allocate 26.00 MiB (GPU 0; 2.00 GiB total capacity; 1004.34 MiB already allocated; 20.88 MiB free; 1.01 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

训练时读取的图像数据太大，超出了GPU的显存

调小batch