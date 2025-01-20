import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
class EvINRModel(nn.Module):
    def __init__(self, H=180, W=240, recon_colors=False,**kargs):
        super().__init__()
        self.recon_colors = recon_colors
        self.d_output = H * W * 3 if recon_colors else H * W

        #parameter of embedings
        self.positionencoder = PositionalEncoding(pe_embed = kargs['pe_embed'])
        
        self.embed_length = self.positionencoder.embed_length

        #parameter of MLP and Conv
        self.net = Generator_plus_plus(embed_length=self.embed_length, stem_dim_num=kargs['stem_dim_num'], fc_hw_dim=kargs['fc_hw_dim'], expansion=kargs['expansion'], 
        reduction=kargs['reduction'],act =kargs['act'] , stride_list=kargs['stride_list'],  lower_width=kargs['lower_width'])
        
        
        self.H, self.W = H, W
        
    def forward(self, timestamps):
        PE = self.positionencoder(timestamps)
        log_intensity_preds = self.net(PE)
        #print(log_intensity_preds.shape)
        if self.recon_colors:
            log_intensity_preds = log_intensity_preds.reshape(-1, self.H, self.W, 3)
        else:
            log_intensity_preds = log_intensity_preds.reshape(-1, self.H, self.W, 1)
        return log_intensity_preds
     
    def get_losses(self, log_intensity_preds, event_frames):
        # temporal supervision to solve the event generation equation
        event_frame_preds = log_intensity_preds[1:] - log_intensity_preds[0: -1]
        #print(log_intensity_preds.shape)
        temperal_loss = F.mse_loss(event_frame_preds, event_frames[:-1])
        # spatial regularization to reduce noise
        x_grad = log_intensity_preds[:, 1: , :, :] - log_intensity_preds[:, 0:-1, :, :]
        y_grad = log_intensity_preds[:, :, 1: , :] - log_intensity_preds[:, :, 0: -1, :]
        spatial_loss = 0.06 * (
            x_grad.abs().mean() + y_grad.abs().mean() + event_frame_preds.abs().mean()
        )

        # loss term to keep the average intensity of each frame constant
        const_loss = 0.1 * torch.var(
            log_intensity_preds.reshape(log_intensity_preds.shape[0], -1).mean(dim=-1)
        )
        print('temperal_loss:{}'.format(temperal_loss))
        return (temperal_loss + spatial_loss + const_loss)
    
    def tonemapping(self, log_intensity_preds, gamma=0.6):
        intensity_preds = torch.exp(log_intensity_preds).detach()
        # Reinhard tone-mapping
        intensity_preds = (intensity_preds / (1 + intensity_preds)) ** (1 / gamma)
        intensity_preds = intensity_preds.clamp(0, 1)
        return intensity_preds


class Pwconv2d(nn.Module):
    def __init__(self, in_channel=180, out_channel=240):
        super().__init__()
        self.function = nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size=(1,1) ,stride=1,padding = 0, groups=1, bias=True)
    def forward(self, x):
        output = self.function(x)
        return output

class Dwconv2d(nn.Module):
    def __init__(self, in_channel=180):
        super().__init__()
        self.function = nn.Conv2d(in_channels = in_channel, out_channels = in_channel, kernel_size=(3,3) ,stride=1, padding = 1,groups=in_channel, bias=True)
    def forward(self, x):
        output = self.function(x)
        return output

class SCRB(nn.Module):
    def __init__(self, in_channel=180, mid_channel = 360):
        super().__init__()
        self.dwconv = Dwconv2d(in_channel=in_channel)
        self.pwconv1 = Pwconv2d(in_channel = in_channel, out_channel = mid_channel)
        self.pwconv2 = Pwconv2d(in_channel = mid_channel, out_channel = in_channel)
        self.activation = nn.GELU()
    def forward(self, x):
        dw_output = self.dwconv(x)
        output = self.pwconv2(self.activation(self.pwconv1(dw_output))) + x
        return output

class UB(nn.Module):
    def __init__(self, in_channel=180,out_channel = 360, stride = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_channel, out_channels = out_channel*stride*stride, kernel_size=(3,3) ,stride=1, padding = 1,groups=in_channel, bias=True)
        self.up_scale = nn.PixelShuffle(stride)
        self.activation = nn.SiLU(inplace=True)
    def forward(self, x):
        output = self.up_scale(self.activation(self.conv(x)))
        return output



class Nerv_plus_plus_block(nn.Module):
    def __init__(self, in_channel=180, SCRB1_in_channel = 240, stride = 2 ,middle_channel = 360):
        super().__init__()
        self.pwconv0 = Pwconv2d(in_channel = in_channel, out_channel = SCRB1_in_channel)
        self.srb1 = SCRB(in_channel=SCRB1_in_channel, mid_channel =middle_channel )
        self.ub = UB(in_channel=SCRB1_in_channel,out_channel = SCRB1_in_channel, stride = 2 )
        self.srb2 = SCRB(in_channel=SCRB1_in_channel, mid_channel =middle_channel )
        self.interporation = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
    def forward(self, x):
        y = self.pwconv0(x)
        output = self.interporation(y)+self.srb2(self.ub(self.srb1(y)))
        return output

class Head(nn.Module):
    def __init__(self, in_channel=180):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_channel, out_channels = 1, kernel_size=(3,3) ,stride=1, padding = 1, bias=True)
        #self.activation = nn.SiLU(inplace=True)
    def forward(self, x):
        y = self.conv(x)
        output = y
        return output



class Generator_plus_plus(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        stem_dim, stem_num = [int(x) for x in kargs['stem_dim_num'].split('_')]
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in kargs['fc_hw_dim'].split('_')]
        mlp_dim_list = [kargs['embed_length']] + [stem_dim] * stem_num + [self.fc_h *self.fc_w *self.fc_dim]
        self.stem = MLP(dim_list=mlp_dim_list, act=kargs['act'])
        self.layers = nn.ModuleList()
        ngf = self.fc_dim
        for i, stride in enumerate(kargs['stride_list']):
            if i == 0:
                # expand channel width at first stage
                new_ngf = int(ngf * kargs['expansion'])
            else:
                # change the channel width for each stage
                new_ngf = max(ngf // (1 if stride == 1 else kargs['reduction']), kargs['lower_width'])
            self.layers.append(Nerv_plus_plus_block(in_channel=ngf, SCRB1_in_channel=new_ngf, stride=stride))
            ngf = new_ngf

            head_layer = Head(in_channel = new_ngf) #out_channels=1 if not use RGB
            self.head_layers = head_layer
    def forward(self, input):
        output = self.stem(input)
        output = output.view(output.size(0), self.fc_dim, self.fc_h, self.fc_w)
        for layer in self.layers:
            output = layer(output) 
        img_out = self.head_layers(output)
        return  img_out    



def ActivationLayer(act_type):
    if act_type == 'relu':
        act_layer = nn.ReLU(True)
    elif act_type == 'leaky':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == 'leaky01':
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'relu6':
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'sin':
        act_layer = torch.sin
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        act_layer = nn.Softplus()
    elif act_type == 'hardswish':
        act_layer = nn.Hardswish(inplace=True)
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer


def NormLayer(norm_type, ch_width):    
    if norm_type == 'none':
        norm_layer = nn.Identity()
    elif norm_type == 'bn':
        norm_layer = nn.BatchNorm2d(num_features=ch_width)
    elif norm_type == 'in':
        norm_layer = nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer


class CustomConv(nn.Module):
    def __init__(self, **kargs):
        super(CustomConv, self).__init__()

        ngf, new_ngf, stride = kargs['ngf'], kargs['new_ngf'], kargs['stride']
        self.conv_type = kargs['conv_type']
        if self.conv_type == 'conv':
            self.conv = nn.Conv2d(ngf, new_ngf * stride * stride, 3, 1, 1, bias=kargs['bias'])
            self.up_scale = nn.PixelShuffle(stride)
        elif self.conv_type == 'deconv':
            self.conv = nn.ConvTranspose2d(ngf, new_ngf, stride, stride)
            self.up_scale = nn.Identity()
        elif self.conv_type == 'bilinear':
            self.conv = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
            self.up_scale = nn.Conv2d(ngf, new_ngf, 2*stride+1, 1, stride, bias=kargs['bias'])

    def forward(self, x):
        out = self.conv(x)
        return self.up_scale(out)


def MLP(dim_list, act='relu', bias=True):
    act_fn = ActivationLayer(act)
    fc_list = []
    for i in range(len(dim_list) - 1):
        fc_list += [nn.Linear(dim_list[i], dim_list[i+1], bias=bias), act_fn]
    return nn.Sequential(*fc_list)


class NeRVBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()

        self.conv = CustomConv(ngf=kargs['ngf'], new_ngf=kargs['new_ngf'], stride=kargs['stride'], bias=kargs['bias'], 
            conv_type=kargs['conv_type'])
        self.norm = NormLayer(kargs['norm'], kargs['new_ngf'])
        self.act = ActivationLayer(kargs['act'])

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, pe_embed):
        super(PositionalEncoding, self).__init__()
        self.pe_embed = pe_embed.lower()
        if self.pe_embed == 'none':
            self.embed_length = 1
        else:
            self.lbase, self.levels = [float(x) for x in pe_embed.split('_')]
            self.levels = int(self.levels)
            self.embed_length = 2 * self.levels

    def forward(self, pos):
        if self.pe_embed == 'none':
            return pos[:,None]
        else:
            pe_list = []
            for i in range(self.levels):
                temp_value = pos * self.lbase **(i) * math.pi
                pe_list += [torch.sin(temp_value), torch.cos(temp_value)]
            return torch.stack(pe_list, 1).squeeze(-1)
        
class Generator(nn.Module):
    def __init__(self, **kargs):
        super().__init__()

        stem_dim, stem_num = [int(x) for x in kargs['stem_dim_num'].split('_')]
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in kargs['fc_hw_dim'].split('_')]
        mlp_dim_list = [kargs['embed_length']] + [stem_dim] * stem_num + [self.fc_h *self.fc_w *self.fc_dim]
        self.stem = MLP(dim_list=mlp_dim_list, act=kargs['act'])
        
        # BUILD CONV LAYERS
        self.layers, self.head_layers = [nn.ModuleList() for _ in range(2)]
        ngf = self.fc_dim
        for i, stride in enumerate(kargs['stride_list']):
            if i == 0:
                # expand channel width at first stage
                new_ngf = int(ngf * kargs['expansion'])
            else:
                # change the channel width for each stage
                new_ngf = max(ngf // (1 if stride == 1 else kargs['reduction']), kargs['lower_width'])

            for j in range(kargs['num_blocks']):
                self.layers.append(NeRVBlock(ngf=ngf, new_ngf=new_ngf, stride=1 if j else stride,
                    bias=kargs['bias'], norm=kargs['norm'], act=kargs['act'], conv_type=kargs['conv_type']))
                ngf = new_ngf

            # build head classifier, upscale feature layer, upscale img layer 
            head_layer = [None]
            if kargs['sin_res']:
                if i == len(kargs['stride_list']) - 1:
                    head_layer = nn.Conv2d(ngf, 1, 1, 1, bias=kargs['bias']) #out_channels=1 if not use RGB
                    # head_layer = nn.Conv2d(ngf, 1, 3, 1, 1, bias=kargs['bias']) 
                else:
                    head_layer = None
            else:
                head_layer = nn.Conv2d(ngf, 1, 1, 1, bias=kargs['bias']) #out_channels=1 if not use RGB
                # head_layer = nn.Conv2d(ngf, 1, 3, 1, 1, bias=kargs['bias'])
            self.head_layers.append(head_layer)
        self.sigmoid =kargs['sigmoid']

    def forward(self, input):
        output = self.stem(input)
        output = output.view(output.size(0), self.fc_dim, self.fc_h, self.fc_w)
        for layer, head_layer in zip(self.layers, self.head_layers):
            output = layer(output) 
            if head_layer is not None:
                img_out = head_layer(output)

                

        return  img_out
