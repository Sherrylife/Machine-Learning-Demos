# 模拟设备之间的通信
devices = {
    0: [1, 2, 3, 4],
    1: [5, 6, 7, 8],
    2: [9, 10, 11, 12],
    3: [13, 14, 15, 16],
}

def reduce_scatter(devices):
    # 归约操作：逐元素求和
    reduced_data = [0] * len(devices[0])
    for data in devices.values():
        for i, val in enumerate(data):
            reduced_data[i] += val

    print("Reduced data:", reduced_data)  # 输出 [28, 32, 36, 40]

    # 分散操作：按奇偶性分块
    scattered_data = {}
    chunk_size = len(reduced_data) // 4  # 每个块的大小为2（因为分2组）
    for device_id in devices:
        # 设备0和2属于组0，设备1和3属于组1
        group = device_id % 4
        start = group * chunk_size
        end = start + chunk_size
        scattered_data[device_id] = reduced_data[start:end]

    return scattered_data

# 执行 reduce-scatter
scattered_data = reduce_scatter(devices)

# 打印结果
for device_id, data in scattered_data.items():
    print(f"Device {device_id} received: {data}")