import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d_Uno(nn.Module):
    def __init__(self, in_codim, out_codim, dim1,modes1 = None):
        super(SpectralConv1d_Uno, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
        dim1 = Default output grid size along x (or 1st dimension of output domain) 
        Ratio of grid size of the input and the output implecitely 
        set the expansion or contraction farctor along each dimension of the domain.
        modes1 = Number of fourier modes to consider for the integral operator.
                Number of modes must be compatibale with the input grid size 
                and desired output grid size.
                i.e., modes1 <= min( dim1/2, input_dim1/2). 
                Here "input_dim1" is the grid size along x axis (or first dimension) of the input domain.
        in_codim = Input co-domian dimension
        out_codim = output co-domain dimension
        """
        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        self.dim1 = dim1 #output dimensions
        if modes1 is not None:
            self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        else:
            self.modes1 = dim1//2

        self.scale = (1 / (2*in_codim))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, dim1 = None):
        """
        input shape = (batch, in_codim, input_dim1)
        output shape = (batch, out_codim, dim1)
        """
        if dim1 is not None:
            self.dim1 = dim1
        batchsize = x.shape[0]

        x_dtype = x.dtype
        x = x.to(torch.float32)
        x_ft = torch.fft.rfft(x, norm = 'forward')

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  self.dim1//2 + 1 , dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=self.dim1, norm = 'forward')
        x = x.to(x_dtype)
        return x

class pointwise_op_1D(nn.Module):
    """
    All variables are consistent with the SpectralConv1d_Uno class.
    """
    def __init__(self, in_codim, out_codim,dim1):
        super(pointwise_op_1D,self).__init__()
        self.conv = nn.Conv1d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)

    def forward(self,x, dim1 = None):
        if dim1 is None:
            dim1 = self.dim1
        x_out = self.conv(x)

        x_out = torch.nn.functional.interpolate(x_out, size = dim1,mode = 'linear',align_corners=True, antialias= True)
        return x_out

class OperatorBlock_1D(nn.Module):
    """
    Normalize = if true performs InstanceNorm1d on the output.
    Non_Lin = if true, applies point wise nonlinearity.
    All other variables are consistent with the SpectralConv1d_Uno class.
    """
    def __init__(self, in_codim, out_codim,dim1,modes1, Normalize = True,Non_Lin = True):
        super(OperatorBlock_1D,self).__init__()
        self.conv = SpectralConv1d_Uno(in_codim, out_codim, dim1,modes1)
        self.w = pointwise_op_1D(in_codim, out_codim, dim1)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm1d(int(out_codim),affine=True)


    def forward(self,x, dim1 = None):
        """
        input shape = (batch, in_codim, input_dim1)
        output shape = (batch, out_codim, dim1)
        """
        x1_out = self.conv(x,dim1)
        x2_out = self.w(x,dim1)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out

class SpectralConv2d_Uno(nn.Module):           #NEW
    def __init__(self, in_codim, out_codim, dim1, dim2,
                 modes1=None, modes2=None,
                 dropout_p=0):
        super(SpectralConv2d_Uno, self).__init__()

        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        self.dim1 = dim1
        self.dim2 = dim2

        self.use_dropout = dropout_p > 0
        if self.use_dropout:
            self.dropout = nn.Dropout2d(dropout_p)

        if modes1 is not None:
            self.modes1 = modes1
            self.modes2 = modes2
        else:
            self.modes1 = dim1 // 2 - 1
            self.modes2 = dim2 // 2

        self.scale = (1 / (2 * in_codim)) ** 0.5
        self.weights1 = nn.Parameter(torch.view_as_real(
            self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat)
        ))
        self.weights2 = nn.Parameter(torch.view_as_real(
            self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat)
        ))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, dim1=None, dim2=None):

        if dim1 is not None:
            self.dim1 = dim1
            self.dim2 = dim2

        batchsize = x.shape[0]

        # Fourier transform
        x_dtype = x.dtype
        x = x.to(torch.float32)
        x_ft = torch.fft.rfft2(x, norm='forward')

        # Output spectrum
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            self.dim1,
            self.dim2 // 2 + 1,
            device=x.device,
            dtype=torch.cfloat
        )

        # Convert weights back to complex
        weights1 = torch.view_as_complex(self.weights1)
        weights2 = torch.view_as_complex(self.weights2)

        # Apply multiplications on each Fourier block
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], weights1)

        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], weights2)

        # Back to spatial domain
        x = torch.fft.irfft2(out_ft, s=(self.dim1, self.dim2), norm='forward')
        x = x.to(x_dtype)

        # 🔥 Dropout applied HERE (safe!)
        if self.use_dropout:
            x = self.dropout(x)

        return x




# class SpectralConv2d_Uno(nn.Module):
#     def __init__(self, in_codim, out_codim, dim1, dim2,modes1 = None, modes2 = None):
#         super(SpectralConv2d_Uno, self).__init__()

#         """
#         2D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
#         dim1 = Default output grid size along x (or 1st dimension of output domain) 
#         dim2 = Default output grid size along y ( or 2nd dimension of output domain)
#         Ratio of grid size of the input and the output implecitely 
#         set the expansion or contraction farctor along each dimension.
#         modes1, modes2 = Number of fourier modes to consider for the ontegral operator
#                         Number of modes must be compatibale with the input grid size 
#                         and desired output grid size.
#                         i.e., modes1 <= min( dim1/2, input_dim1/2). 
#                         Here "input_dim1" is the grid size along x axis (or first dimension) of the input domain.
#                         Other modes also the have same constrain.
#         in_codim = Input co-domian dimension
#         out_codim = output co-domain dimension
#         """

#         in_codim = int(in_codim)
#         out_codim = int(out_codim)
#         self.in_channels = in_codim
#         self.out_channels = out_codim
#         self.dim1 = dim1 
#         self.dim2 = dim2
#         if modes1 is not None:
#             self.modes1 = modes1
#             self.modes2 = modes2
#         else:
#             self.modes1 = dim1//2-1 
#             self.modes2 = dim2//2 
#         self.scale = (1 / (2*in_codim))**(1.0/2.0)
#         self.weights1 = nn.Parameter(self.scale * (torch.randn(in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat)))
#         self.weights1 = nn.Parameter(torch.view_as_real(self.weights1)) # Matt
#         self.weights2 = nn.Parameter(self.scale * (torch.randn(in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat)))
#         self.weights2 = nn.Parameter(torch.view_as_real(self.weights2)) # Matt

#     # Complex multiplication
#     def compl_mul2d(self, input, weights):

#         #print(f"input.shape = {input.shape}, weights.shape = {weights.shape}")
#         return torch.einsum("bixy,ioxy->boxy", input, weights)

#     def forward(self, x, dim1 = None,dim2 = None):
#         if dim1 is not None:
#             self.dim1 = dim1
#             self.dim2 = dim2
#         batchsize = x.shape[0]
#         #Compute Fourier coeffcients up to factor of e^(- something constant)
#         x_ft = torch.fft.rfft2(x, norm = 'forward')

#         # Multiply relevant Fourier modes
#         out_ft = torch.zeros(batchsize, self.out_channels,  self.dim1, self.dim2//2 + 1 , dtype=torch.cfloat, device=x.device)
#         #print(f"out_ft.shape = {out_ft.shape}, self.out_channels = {self.out_channels}, self.dim1 = {self.dim1}, self.dim2//2+1 = {self.dim2//2+1}")
#         #print(f"out_ft[:, :, :self.modes1, :self.modes2].shape = {out_ft[:, :, :self.modes1, :self.modes2].shape}")

#         #print(f"x_ft.shape = {x_ft.shape}, self.modes1 = {self.modes1}, self.modes2 = {self.modes2}")
#         #print(f"self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1).shape = {self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1).shape}")


#         weights1 = torch.view_as_complex(self.weights1) # Matt
#         weights2 = torch.view_as_complex(self.weights2) # Matt

#         out_ft[:, :, :self.modes1, :self.modes2] = \
#                 self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], weights1) # Matt: self.weights1->weights1
#         out_ft[:, :, -self.modes1:, :self.modes2] = \
#                 self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], weights2) # Matt: self.weights2->weights2

#         #Return to physical space
#         x = torch.fft.irfft2(out_ft, s=(self.dim1, self.dim2),norm = 'forward')
#         return x

class pointwise_op_2D(nn.Module):
    """ 
    dim1 = Default output grid size along x (or 1st dimension) 
    dim2 = Default output grid size along y ( or 2nd dimension)
    in_codim = Input co-domian dimension
    out_codim = output co-domain dimension
    """
    def __init__(self, in_codim, out_codim,dim1, dim2):
        super(pointwise_op_2D,self).__init__()
        self.conv = nn.Conv2d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)

    def forward(self,x, dim1 = None, dim2 = None):
        """
        input shape = (batch, in_codim, input_dim1,input_dim2)
        output shape = (batch, out_codim, dim1,dim2)
        """
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
        x_out = self.conv(x)

        #ft = torch.fft.rfft2(x_out)
        #ft_u = torch.zeros_like(ft)
        #ft_u[:dim1//2-1,:dim2//2-1] = ft[:dim1//2-1,:dim2//2-1]
        #ft_u[-(dim1//2-1):,:dim2//2-1] = ft[-(dim1//2-1):,:dim2//2-1]
        #x_out = torch.fft.irfft2(ft_u)
        
        x_out = torch.nn.functional.interpolate(x_out, size = (dim1, dim2),mode = 'bicubic',align_corners=True, antialias=True)
        return x_out

# class OperatorBlock_2D(nn.Module):
#     """
#     Normalize = if true performs InstanceNorm2d on the output.
#     Non_Lin = if true, applies point wise nonlinearity.
#     dropout = probability of an element to be zeroed.
#     All other variables are consistent with the SpectralConv2d_Uno class.
#     """
#     def __init__(self, in_codim, out_codim, dim1, dim2, modes1, modes2, Normalize = True, Non_Lin = True, dropout=0.15):
#         super(OperatorBlock_2D,self).__init__()
#         self.conv = SpectralConv2d_Uno(in_codim, out_codim, dim1,dim2,modes1,modes2)
#         self.w = pointwise_op_2D(in_codim, out_codim, dim1,dim2)
#         self.normalize = Normalize
#         self.non_lin = Non_Lin
#         if Normalize:
#             self.normalize_layer = torch.nn.InstanceNorm2d(int(out_codim),affine=True)
        
#         # --- CHANGE: Initialize Dropout ---
#         self.dropout = nn.Dropout(dropout) 

#     def forward(self,x, dim1 = None, dim2 = None):
#         """
#         input shape = (batch, in_codim, input_dim1,input_dim2)
#         output shape = (batch, out_codim, dim1,dim2)
#         """
#         x1_out = self.conv(x,dim1,dim2)
#         x2_out = self.w(x,dim1,dim2)
#         x_out = x1_out + x2_out
        
#         if self.normalize:
#             x_out = self.normalize_layer(x_out)
        
#         if self.non_lin:
#             x_out = F.gelu(x_out)
            
#         # --- CHANGE: Apply Dropout ---
#         x_out = self.dropout(x_out)
        
#         return x_out

class OperatorBlock_2D(nn.Module):
    """
    Normalize = if true performs InstanceNorm2d on the output.
    Non_Lin = if true, applies point wise nonlinearity.
    All other variables are consistent with the SpectralConv2d_Uno class.
    """
    def __init__(self, in_codim, out_codim,dim1, dim2,modes1,modes2, Normalize = True, Non_Lin = True):
        super(OperatorBlock_2D,self).__init__()
        self.conv = SpectralConv2d_Uno(in_codim, out_codim, dim1,dim2,modes1,modes2)
        self.w = pointwise_op_2D(in_codim, out_codim, dim1,dim2)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm2d(int(out_codim),affine=True)


    def forward(self,x, dim1 = None, dim2 = None):
        """
        input shape = (batch, in_codim, input_dim1,input_dim2)
        output shape = (batch, out_codim, dim1,dim2)
        """
        #print(f"dim1 = {dim1}, dim2 = {dim2}")
        x1_out = self.conv(x,dim1,dim2)
        x2_out = self.w(x,dim1,dim2)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out

class SpectralConv3d_Uno(nn.Module):
    def __init__(self, in_codim, out_codim,dim1,dim2,dim3, modes1=None, modes2=None, modes3=None):
        super(SpectralConv3d_Uno, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
        dim1 = Default output grid size along x (or 1st dimension of output domain) 
        dim2 = Default output grid size along y ( or 2nd dimension of output domain)
        dim3 = Default output grid size along time t ( or 3rd dimension of output domain)
        Ratio of grid size of the input and output grid size (dim1,dim2,dim3) implecitely 
        set the expansion or contraction farctor along each dimension.
        modes1, modes2, modes3 = Number of fourier modes to consider for the ontegral operator
                                Number of modes must be compatibale with the input grid size 
                                and desired output grid size.
                                i.e., modes1 <= min( dim1/2, input_dim1/2).
                                      modes2 <= min( dim2/2, input_dim2/2)
                                Here input_dim1, input_dim2 are respectively the grid size along 
                                x axis and y axis (or first dimension and second dimension) of the input domain.
                                Other modes also have the same constrain.
        in_codim = Input co-domian dimension
        out_codim = output co-domain dimension   
        """
        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        if modes1 is not None:
            self.modes1 = modes1 
            self.modes2 = modes2
            self.modes3 = modes3 
        else:
            self.modes1 = dim1 
            self.modes2 = dim2
            self.modes3 = dim3//2+1

        self.scale = (1 / (2*in_codim))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):

        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x, dim1 = None,dim2=None,dim3=None):
        """
        dim1,dim2,dim3 are the output grid size along (x,y,t)
        input shape = (batch, in_codim, input_dim1, input_dim2, input_dim3)
        output shape = (batch, out_codim, dim1,dim2,dim3)
        """
        if dim1 is not None:
            self.dim1 = dim1
            self.dim2 = dim2
            self.dim3 = dim3   

        batchsize = x.shape[0]

        x_dtype = x.dtype
        x = x.to(torch.float32)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1], norm = 'forward')

        out_ft = torch.zeros(batchsize, self.out_channels, self.dim1, self.dim2, self.dim3//2 + 1, dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(self.dim1, self.dim2, self.dim3), norm = 'forward')
        x = x.to(x_dtype)
        return x

class pointwise_op_3D(nn.Module):
    def __init__(self, in_codim, out_codim,dim1, dim2,dim3):
        super(pointwise_op_3D,self).__init__()
        self.conv = nn.Conv3d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.dim3 = int(dim3)

    def forward(self,x, dim1 = None, dim2 = None, dim3 = None):
        """
        dim1,dim2,dim3 are the output dimensions (x,y,t)
        """
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
            dim3 = self.dim3
        x_out = self.conv(x)

        x_out_dtype = x_out.dtype
        x_out = x_out.to(torch.float32)
        ft = torch.fft.rfftn(x_out,dim=[-3,-2,-1])
        ft_u = torch.zeros_like(ft)
        ft_u[:, :, :(dim1//2), :(dim2//2), :(dim3//2)] = ft[:, :, :(dim1//2), :(dim2//2), :(dim3//2)]
        ft_u[:, :, -(dim1//2):, :(dim2//2), :(dim3//2)] = ft[:, :, -(dim1//2):, :(dim2//2), :(dim3//2)]
        ft_u[:, :, :(dim1//2), -(dim2//2):, :(dim3//2)] = ft[:, :, :(dim1//2), -(dim2//2):, :(dim3//2)]
        ft_u[:, :, -(dim1//2):, -(dim2//2):, :(dim3//2)] = ft[:, :, -(dim1//2):, -(dim2//2):, :(dim3//2)]
        
        x_out = torch.fft.irfftn(ft_u, s=(dim1, dim2, dim3))
        x_out = x_out.to(x_out_dtype)

        x_out = torch.nn.functional.interpolate(x_out, size = (dim1, dim2,dim3),mode = 'trilinear',align_corners=True)
        return x_out

class OperatorBlock_3D(nn.Module):
    """
    Normalize = if true performs InstanceNorm3d on the output.
    Non_Lin = if true, applies point wise nonlinearity.
    All other variables are consistent with the SpectralConv3d_Uno class.
    """
    def __init__(self, in_codim, out_codim,dim1, dim2,dim3,modes1,modes2,modes3, Normalize = True,Non_Lin = True):
        super(OperatorBlock_3D,self).__init__()
        self.conv = SpectralConv3d_Uno(in_codim, out_codim, dim1,dim2,dim3,modes1,modes2,modes3)
        self.w = pointwise_op_3D(in_codim, out_codim, dim1,dim2,dim3)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm3d(int(out_codim),affine=True)


    def forward(self,x, dim1 = None, dim2 = None, dim3 = None):
        """
        input shape = (batch, in_codim, input_dim1, input_dim2, input_dim3)
        output shape = (batch, out_codim, dim1,dim2,dim3)
        """
        x1_out = self.conv(x,dim1,dim2,dim3)
        x2_out = self.w(x,dim1,dim2,dim3)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out

class HybridOperatorBlock_2D(nn.Module):
    """
    Hybrid block that adds a parallel local spatial convolution (3x3) to the
    Spectral Convolution to act as an anti-ringing spatial filter.
    """
    def __init__(self, in_codim, out_codim,dim1, dim2,modes1,modes2, Normalize = True, Non_Lin = True):
        super(HybridOperatorBlock_2D,self).__init__()
        self.conv = SpectralConv2d_Uno(in_codim, out_codim, dim1,dim2,modes1,modes2)
        self.w = pointwise_op_2D(in_codim, out_codim, dim1,dim2)
        
        # Parallel 3x3 local convolution for anti-ringing
        self.local_conv = nn.Conv2d(int(in_codim), int(out_codim), kernel_size=3, padding=1, padding_mode='circular')
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm2d(int(out_codim),affine=True)


    def forward(self,x, dim1 = None, dim2 = None):
        """
        input shape = (batch, in_codim, input_dim1,input_dim2)
        output shape = (batch, out_codim, dim1,dim2)
        """
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
            
        x1_out = self.conv(x,dim1,dim2)
        x2_out = self.w(x,dim1,dim2)
        
        # Local branch
        x3_out = self.local_conv(x)
        if x3_out.shape[-2:] != (dim1, dim2):
            x3_out = torch.nn.functional.interpolate(x3_out, size=(dim1, dim2), mode='bicubic', align_corners=True, antialias=True)
            
        x_out = x1_out + x2_out + x3_out
        
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out
