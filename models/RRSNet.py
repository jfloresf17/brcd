import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class RRSNet(nn.Module):
    """
    PyTorch Module for RRSNet.
    Now x4 is only supported.

    Parameters
    ---
    ngf : int, optional
        the number of filterd of generator.
    n_blocks : int, optional
        the number of residual blocks for each module.
    """
    def __init__(self, ngf=64, n_blocks=16, groups=8):
        super(RRSNet, self).__init__()
        self.get_g_nopadding = Get_gradient_nopadding()
        self.Encoder = Encoder(3, nf=ngf)
        self.Encoder_grad = Encoder(3, nf=int(ngf/4))
        self.gpcd_align = GPCD_Align(nf=ngf+int(ngf/4), nf_out=ngf, groups=groups)
        self.content_extractor = ContentExtractor(ngf, n_blocks)
        self.texture_transfer = TextureTransfer(ngf, n_blocks)      


    def forward(self, LR, LR_UX4, Ref,Ref_DUX4, weights=None):
        LR_UX4_grad = self.get_g_nopadding(LR_UX4)
        Ref_DUX4_grad = self.get_g_nopadding(Ref_DUX4)
        Ref_grad = self.get_g_nopadding(Ref)


        LR_conv1, LR_conv2, LR_conv3 = self.Encoder(LR_UX4)        
        HR1_conv1, HR1_conv2, HR1_conv3 = self.Encoder(Ref_DUX4)
        HR2_conv1, HR2_conv2, HR2_conv3 = self.Encoder(Ref)
        
        #grad
        LR_conv1_grad, LR_conv2_grad, LR_conv3_grad = self.Encoder_grad(LR_UX4_grad)
        HR1_conv1_grad, HR1_conv2_grad, HR1_conv3_grad = self.Encoder_grad(Ref_DUX4_grad)
        HR2_conv1_grad, HR2_conv2_grad, HR2_conv3_grad = self.Encoder_grad(Ref_grad)

        LR_conv1 = torch.cat([LR_conv1,LR_conv1_grad], dim=1)
        LR_conv2 = torch.cat([LR_conv2,LR_conv2_grad], dim=1)
        LR_conv3 = torch.cat([LR_conv3,LR_conv3_grad], dim=1)
        HR2_conv1 = torch.cat([HR2_conv1,HR2_conv1_grad], dim=1)
        HR2_conv2 = torch.cat([HR2_conv2,HR2_conv2_grad], dim=1)
        HR2_conv3 = torch.cat([HR2_conv3,HR2_conv3_grad], dim=1)
        HR1_conv1 = torch.cat([HR1_conv1,HR1_conv1_grad], dim=1)
        HR1_conv2 = torch.cat([HR1_conv2,HR1_conv2_grad], dim=1)
        HR1_conv3 = torch.cat([HR1_conv3,HR1_conv3_grad], dim=1)

        LR_fea_l = [LR_conv1, LR_conv2, LR_conv3]
        Ref_use_fea_l = [HR2_conv1, HR2_conv2, HR2_conv3]
        Ref_fea_l = [HR1_conv1, HR1_conv2, HR1_conv3]

        Ref_conv1, Ref_conv2, Ref_conv3 = self.gpcd_align(Ref_use_fea_l,Ref_fea_l,LR_fea_l)
        maps = [Ref_conv1, Ref_conv2, Ref_conv3]

        base = F.interpolate(LR, None, 4, 'bilinear', False)
        upscale_plain, content_feat = self.content_extractor(LR)

        upscale_rrsnet = self.texture_transfer(
                    content_feat, maps)
        return upscale_rrsnet + base


class DCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False,
                 deformable_groups=1):  # Agregar deformable groups
        super(DCN, self).__init__()

        # Parametrización
        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups

        # Offset convolution (generar offsets para deformable convolutions)
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size[0] * kernel_size[1] * deformable_groups,  # Ajustar para deformable groups
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True
        )
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        # Modulator convolution (para máscaras en deformable convolutions)
        self.modulator_conv = nn.Conv2d(
            in_channels,
            1 * kernel_size[0] * kernel_size[1] * deformable_groups,  # Ajustar para deformable groups
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True
        )
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        # Regular convolution (operación base de convolución)
        self.regular_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias
        )

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, dim=1)  # Concatenar tensores en la dimensión de los canales
        
        x = nn.Conv2d(x.shape[1], self.offset_conv.in_channels, kernel_size=1).to(x.device)(x)

        # Calcular offset y modulator
        offset = self.offset_conv(x)  # Generar los offsets para deformable convolutions
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))  # Generar la máscara (valores entre 0 y 2)

        # Aplicar deformable convolution
        x = torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=modulator,
            stride=self.stride,
            dilation=self.dilation
        )
        return x


class GPCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=64,nf_out=64, groups=8):
        super(GPCD_Align, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)   
        self.L3_fea_conv = nn.Conv2d(nf, nf_out, 3, 1, 1, bias=True)
       
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L2_fea_conv = nn.Conv2d(nf + nf_out, nf_out, 3, 1, 1, bias=True)  # concat for fea
        
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L1_fea_conv = nn.Conv2d(nf + nf_out, nf, 3, 1, 1, bias=True)  # concat for fea
        
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)

        self.cas_fea_conv = nn.Conv2d(nf, nf_out, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, Ref_use_fea_l, Ref_fea_l, LR_fea_l):
        '''
        Ref_use_fea_l, Ref_fea_l, LR_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L3
        L3_offset = torch.cat([Ref_fea_l[2], LR_fea_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))        
        L3_fea = self.L3_dcnpack([Ref_use_fea_l[2], L3_offset])
        L3_fea_output = self.lrelu(self.L3_fea_conv(L3_fea))
        
        # L2
        L2_offset = torch.cat([Ref_fea_l[1], LR_fea_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack([Ref_use_fea_l[1], L2_offset])
        L3_fea = F.interpolate(L3_fea_output, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea_output = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))

        # L1
        L1_offset = torch.cat([Ref_fea_l[0], LR_fea_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_dcnpack([Ref_use_fea_l[0], L1_offset])
        L2_fea = F.interpolate(L2_fea_output, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
        
        # Cascading
        offset = torch.cat([L1_fea, LR_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea_output = self.cas_dcnpack([L1_fea, offset])
        L1_fea_output = self.lrelu(self.cas_fea_conv(L1_fea_output))

        return L1_fea_output, L2_fea_output, L3_fea_output


class ContentExtractor(nn.Module):
    def __init__(self, ngf=64, n_blocks=16):
        super(ContentExtractor, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(3, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True)
        )
        self.body = nn.Sequential(
            *[ResBlock(ngf) for _ in range(n_blocks)],
        )
        self.tail = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        h = self.head(x)
        h = self.body(h) + h
        upscale = self.tail(h)
        return upscale, h


class TextureTransfer(nn.Module):
    def __init__(self, ngf=64, n_blocks=16):
        super(TextureTransfer, self).__init__()

        # for small scale
        self.ram_head_small = RAM()      
        self.head_small = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
        )

        self.body_small = nn.Sequential(
            *[ResBlock(ngf) for _ in range(n_blocks)],
        )
        self.tail_small = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
        )

        # for medium scale
        self.ram_head_medium = RAM()
        self.head_medium = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
        )

        self.body_medium = nn.Sequential(
            *[ResBlock(ngf) for _ in range(n_blocks)],
        )
        self.tail_medium = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
        )

        # for large scale
        self.ram_head_large = RAM()
        self.head_large = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
        )

        self.body_large = nn.Sequential(
            *[ResBlock(ngf) for _ in range(n_blocks)],
        )
        self.tail_large = nn.Sequential(
            nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf // 2, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, maps):
        # small scale
        h = self.ram_head_small(maps[2], x)
        h = torch.cat([x, h], 1)
        h = self.head_small(h)
        h = self.body_small(h) + x
        x = self.tail_small(h)

        # medium scale
        h = self.ram_head_medium(maps[1],x)
        h = torch.cat([x, h], 1)
        h = self.head_medium(h)
        h = self.body_medium(h) + x
        x = self.tail_medium(h)

        # large scale
        h = self.ram_head_large(maps[0],x)
        h = torch.cat([x, h], 1)
        h = self.head_large(h)
        h = self.body_large(h) + x
        x = self.tail_large(h)

        return x


class ResBlock(nn.Module):
    def __init__(self, n_filters=64):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
        )

    def forward(self, x):
        return self.body(x) + x


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)
 
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)


    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim = 1)

        return x

class RAM(nn.Module):
    def __init__(self, nf=64, n_condition=64):
        super(RAM, self).__init__()
        self.mul_conv1 = nn.Conv2d(nf + n_condition, 32, kernel_size=3, stride=1, padding=1)
        self.mul_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)
        self.add_conv1 = nn.Conv2d(nf + n_condition, 32, kernel_size=3, stride=1, padding=1)
        self.add_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, features, conditions):
        cat_input = torch.cat((features, conditions), dim=1)
        mul = torch.sigmoid(self.mul_conv2(self.lrelu(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.lrelu(self.add_conv1(cat_input)))
        return features * mul + add



# 
def conv_activation(in_ch, out_ch , kernel_size = 3, stride = 1, padding = 1, activation = 'relu', init_type = 'w_init_relu'):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding),
        nn.SELU(inplace = True))

class Encoder(nn.Module):
    def __init__(self,in_ch, nf, activation = 'selu', init_type = 'w_init'):
        super(Encoder, self).__init__()
        self.layer_f = conv_activation(in_ch, nf, kernel_size = 5 ,stride = 1,padding = 2, activation = activation, init_type = init_type)
        self.conv1 = conv_activation(nf, nf, kernel_size = 5 ,stride = 1,padding = 2, activation = activation, init_type = init_type)
        self.conv2 = conv_activation(nf, nf, kernel_size = 5 ,stride = 2,padding = 2, activation = activation, init_type = init_type)
        self.conv3 = conv_activation(nf, nf, kernel_size = 5 ,stride = 2,padding = 2, activation = activation, init_type = init_type)

    def forward(self,x):
        layer_f = self.layer_f(x)
        conv1 = self.conv1(layer_f)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        return conv1,conv2,conv3

class ResBlock(nn.Module):
    def __init__(self, n_filters=64):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
        )

    def forward(self, x):
        return self.body(x) + x