
def conv2d_out(x_in, kernel_size, stride=1, padding=0, dilation=1):
    x_out = (x_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    return x_out

def conv_transpose2d_out(x_in, kernel_size, stride=1, padding=0, dilation=1, output_padding=0):
    x_out = (x_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1 
    return x_out

size = conv_transpose2d_out(x_in=50, kernel_size=4, stride=2, padding=1)
print(size)


