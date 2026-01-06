import torch
import torch.nn as nn

# 函数生成指定阶数的Legendre多项式
def legendre_polynomials(x, degree):
    """
    Generate Legendre polynomials up to the given degree.
    :param x: Input tensor of shape (batch_size, input_dim)
    :param degree: Maximum degree of the polynomial
    :return: Tensor of shape (batch_size, input_dim, degree + 1)
    """
    p0 = torch.ones_like(x)  # P_0(x) = 1
    if degree == 0:
        return p0.unsqueeze(-1)
    
    p1 = x  # P_1(x) = x
    if degree == 1:
        return torch.stack([p0, p1], dim=-1)
    
    polys = [p0, p1]
    for n in range(1, degree):
        pn = ((2 * n + 1) * x * polys[-1] - n * polys[-2]) / (n + 1)
        polys.append(pn)
    
    return torch.stack(polys, dim=-1)  

# 定义Legendre多项式版本的KAN层
class LegendreKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        """
        :param input_dim: Dimensionality of the input
        :param output_dim: Dimensionality of the output
        :param degree: Degree of the Legendre polynomial
        """
        super(LegendreKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        # 初始化Legendre多项式的系数参数
        self.legendre_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.legendre_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
    
    def forward(self, x):
        x = torch.tanh(x)

        # 生成Legendre多项式，形状为 (batch_size, input_dim, degree + 1)
        legendre_terms = legendre_polynomials(x, self.degree)

        y = torch.einsum("bid,iod->bo", legendre_terms, self.legendre_coeffs)

        return y.view(-1, self.outdim)

# 示例用法
if __name__ == "__main__":
    input_dim = 3
    output_dim = 2
    degree = 4
    batch_size = 5

    # 创建LegendreKANLayer实例
    layer = LegendreKANLayer(input_dim=input_dim, output_dim=output_dim, degree=degree)
    
    # 随机生成输入数据
    x = torch.randn(batch_size, input_dim)
    
    # 前向传播
    y = layer(x)
    print("Input:", x)
    print("Output:", y)
