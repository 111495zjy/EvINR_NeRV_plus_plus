from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
os.environ['SDL_AUDIODRIVER'] = 'dummy'
from PIL import Image
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
import random
from event_data import EventData
from model import EvINRModel

def config_parser():
    parser = ArgumentParser(description="EvINR")
    parser.add_argument('--exp_name', '-n', type=str, help='Experiment name')
    parser.add_argument('--data_path', '-d', type=str, help='Path of events.npy to train')
    parser.add_argument('--output_dir', '-o', type=str, default='logs', help='Directory to save output')
    parser.add_argument('--t_start', type=float, default=0, help='Start time')
    parser.add_argument('--t_end', type=float, default=28, help='End time')
    parser.add_argument('--H', type=int, default=180, help='Height of frames')
    parser.add_argument('--W', type=int, default=240, help='Width of frames')
    parser.add_argument('--color_event', action='store_true', default=False, help='Whether to use color event')
    parser.add_argument('--event_thresh', type=float, default=1, help='Event activation threshold')
    parser.add_argument('--train_resolution', type=int, default=50, help='Number of training frames')
    parser.add_argument('--val_resolution', type=int, default=50, help='Number of validation frames')
    parser.add_argument('--no_c2f', action='store_true', default=False, help='Whether to use coarse-to-fine training')
    parser.add_argument('--iters', type=int, default=5000, help='Training iterations')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--net_layers', type=int, default=3, help='Number of layers in the network')
    parser.add_argument('--net_width', type=int, default=512, help='Hidden dimension of the network')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--stem_dim_num', type=str, default='512_1', help='hidden dimension and length')
    parser.add_argument('--fc_hw_dim', type=str, default='45_60_64', help='out size (h,w) for mlp')
    parser.add_argument('--expansion', type=float, default=1, help='channel expansion from fc to conv')
    parser.add_argument('--reduction', type=int, default=2)
    parser.add_argument('--num_blocks', type=int, default=1)
    parser.add_argument('--norm', default='none', type=str, help='norm layer for generator', choices=['none', 'bn', 'in'])
    parser.add_argument('--act', type=str, default='gelu', help='activation to use', choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'])
    parser.add_argument("--conv_type", default='conv', type=str,  help='upscale methods, can add bilinear and deconvolution methods', choices=['conv', 'deconv', 'bilinear'])
    parser.add_argument('--strides', type=int, nargs='+', default=[2,2], help='strides list')
    parser.add_argument("--single_res", action='store_true', help='single resolution,  added to suffix!!!!')
    parser.add_argument('--lower_width', type=int, default=32, help='lowest channel width for output feature maps')
    parser.add_argument('--sigmoid', action='store_true', help='using sigmoid for output prediction')
    parser.add_argument('--embed', type=str, default='1.25_40', help='base value/embed length for position encoding')
    parser.add_argument('--model_dir', type=str, default='/content/NeRV_based_EvINR/models', help='saving path')


    return parser

def main(args):
    events = EventData(
        args.data_path, args.t_start, args.t_end, args.H, args.W, args.color_event, args.event_thresh, args.device)
    model = EvINRModel(
         H=180, W=240, recon_colors=args.color_event,stem_dim_num=args.stem_dim_num, fc_hw_dim=args.fc_hw_dim, expansion=args.expansion, 
        act=args.act, reduction=args.reduction, stride_list=args.strides, lower_width=args.lower_width, pe_embed = args.embed).to(args.device)
    optimizer = torch.optim.AdamW(params=model.net.parameters(), lr=3e-4)

    writer = SummaryWriter(os.path.join(args.output_dir, args.exp_name))
    print(f'Start training ...')
    events.stack_event_frames(args.train_resolution)
    for i_iter in trange(1, args.iters + 1):
        #events = EventData(
          #args.data_path, args.t_start, args.t_end, args.H, args.W, args.color_event, args.event_thresh, args.device)
        optimizer.zero_grad()
        
        #events.stack_event_frames(30+random.randint(1, 30))
        log_intensity_preds = model(events.timestamps)
        loss = model.get_losses(log_intensity_preds, events.event_frames)

        loss.backward()
        optimizer.step()
        
        if i_iter % args.log_interval == 0:
            tqdm.write(f'iter {i_iter}, loss {loss.item():.4f}')
            writer.add_scalar('loss', loss.item(), i_iter)

        if not args.no_c2f and i_iter == (args.iters // 2):
            events.stack_event_frames(60)
        if i_iter==1000:
          events.stack_event_frames(65)
        if i_iter==2000: 
          events.stack_event_frames(70)


    with torch.no_grad():
        val_timestamps = torch.linspace(-1, 1, args.val_resolution).to(args.device).reshape(-1, 1)
        log_intensity_preds = model(val_timestamps)
        intensity_preds = model.tonemapping(log_intensity_preds).squeeze(-1)
        for i in range(0, intensity_preds.shape[0]):
            intensity1 = intensity_preds[i].cpu().detach().numpy()
            image_data = (intensity1*255).astype(np.uint8)

            # 将 NumPy 数组转换为 PIL 图像对象
            image = Image.fromarray(image_data)
            output_path = os.path.join('/content/EvINR_NeRV_plus_plus/logs', 'output_image_{}.png'.format(i))
            image.save(output_path)


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    main(args)
