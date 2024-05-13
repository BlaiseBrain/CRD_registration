import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VoxelMorph(nn.Module):
    def __init__(self, flow_multiplier=1.):
        super(VoxelMorph, self).__init__()
        self.flow_multiplier = flow_multiplier
        #  encoder
        self.conv1 = nn.Conv3d(2, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1)


        #  decoder
        self.decode5 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.decode4 = nn.Conv3d(32 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.decode3 = nn.Conv3d(32 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.decode2 = nn.Conv3d(32 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.decode2_1 = nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1)
        self.decode1 = nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1)

        #  fuser
        self.fuse5 = nn.Conv3d(32 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.fuse4 = nn.Conv3d(32 + 32 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.fuse3 = nn.Conv3d(32 + 16 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.fuse2 = nn.Conv3d(32 + 16 + 16, 32, kernel_size=3, stride=1, padding=1)
        self.fuse1 = nn.Conv3d(32 + 16, 16, kernel_size=3, stride=1, padding=1)
        self.flow_f = nn.Conv3d(16, 3, kernel_size=3, stride=1, padding=1)

        self.flow = nn.Conv3d(16, 3, kernel_size=3, stride=1, padding=1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, image1, image2, Hinput=False):
        concatImgs = torch.cat([image1, image2], 1)

        ###########################Teacher###########################

        encode_1x = self.lrelu(self.conv1(concatImgs))  # 1 -> 1
        encode_2x = self.lrelu(self.conv2(encode_1x))  # 1 -> 1/2
        encode_4x = self.lrelu(self.conv3(encode_2x))  # 1/2 -> 1/4
        encode_8x = self.lrelu(self.conv4(encode_4x))  # 1/4 -> 1/8
        encode_16x = self.lrelu(self.conv5(encode_8x))  # 1/8 -> 1/16

        _, _, d, h, w = encode_8x.shape
        decode_16x = self.lrelu(self.decode5(encode_16x))
        decode_8x_up = F.interpolate(decode_16x, size=(d, h, w))
        decode_8x_concat = torch.cat([decode_8x_up, encode_8x], 1)  # 1/16 -> 1/8

        _, _, d, h, w = encode_4x.shape
        decode_8x = self.lrelu(self.decode4(decode_8x_concat))
        decode_4x_up = F.interpolate(decode_8x, size=(d, h, w))
        decode_4x_concat = torch.cat([decode_4x_up, encode_4x], 1)  # 1/8 -> 1/4

        _, _, d, h, w = encode_2x.shape
        decode_4x = self.lrelu(self.decode3(decode_4x_concat))
        decode_2x_up = F.interpolate(decode_4x, size=(d, h, w))
        decode_2x_concat = torch.cat([decode_2x_up, encode_2x], 1)  # 1/4 -> 1/2

        _, _, d, h, w = encode_1x.shape
        decode_2x = self.lrelu(self.decode2(decode_2x_concat))  # 1/2 -> 1/2
        decode_2x = self.lrelu(self.decode2_1(decode_2x))

        decode_1x_up = F.interpolate(decode_2x, size=(d, h, w))
        decode_1x_concat = torch.cat([decode_1x_up, encode_1x], 1)  # 1/2 -> 1
        decode_1x = self.lrelu(self.decode1(decode_1x_concat))

        # net = self.flow(decode_1x)
        # flow = net

        ###########################Teacher###########################



        ###########################Student###########################

        _, _, d, h, w = concatImgs.shape
        concatImgs_s = F.interpolate(concatImgs, size=(int(d/2), int(h/2), int(w/2)), mode='trilinear', align_corners=True)
        encode_1x_s = self.lrelu(self.conv1(concatImgs_s))  # 1 -> 1
        encode_2x_s = self.lrelu(self.conv2(encode_1x_s))  # 1 -> 1/2
        encode_4x_s = self.lrelu(self.conv3(encode_2x_s))  # 1/2 -> 1/4
        encode_8x_s = self.lrelu(self.conv4(encode_4x_s))  # 1/4 -> 1/8
        encode_16x_s = self.lrelu(self.conv5(encode_8x_s))  # 1/8 -> 1/16

        _, _, d, h, w = encode_8x_s.shape
        decode_16x_s = self.lrelu(self.decode5(encode_16x_s))
        decode_8x_up_s = F.interpolate(decode_16x_s, size=(d, h, w))
        decode_8x_concat_s = torch.cat([decode_8x_up_s, encode_8x_s], 1)  # 1/16 -> 1/8

        _, _, d, h, w = encode_4x_s.shape
        decode_8x_s = self.lrelu(self.decode4(decode_8x_concat_s))
        decode_4x_up_s = F.interpolate(decode_8x_s, size=(d, h, w))
        decode_4x_concat_s = torch.cat([decode_4x_up_s, encode_4x_s], 1)  # 1/8 -> 1/4

        _, _, d, h, w = encode_2x_s.shape
        decode_4x_s = self.lrelu(self.decode3(decode_4x_concat_s))
        decode_2x_up_s = F.interpolate(decode_4x_s, size=(d, h, w))
        decode_2x_concat_s = torch.cat([decode_2x_up_s, encode_2x_s], 1)  # 1/4 -> 1/2

        _, _, d, h, w = encode_1x_s.shape
        decode_2x_s = self.lrelu(self.decode2(decode_2x_concat_s))  # 1/2 -> 1/2
        decode_2x_s = self.lrelu(self.decode2_1(decode_2x_s))

        decode_1x_up_s = F.interpolate(decode_2x_s, size=(d, h, w))
        decode_1x_concat_s = torch.cat([decode_1x_up_s, encode_1x_s], 1)  # 1/2 -> 1
        decode_1x_s = self.lrelu(self.decode1(decode_1x_concat_s))

        # net_s = self.flow(decode_1x_s)
        # _, _, d, h, w = concatImgs.shape
        # flow_s = F.interpolate(net_s, size=(d, h, w), mode='trilinear', align_corners=True) * 2.0

        ###########################Student###########################


        #############################Fuse############################

        ts_decode_16x = torch.cat([decode_8x_s, decode_16x], 1)  # 32 + 32
        ts_decode_8x = torch.cat([decode_4x_s, decode_8x], 1)  # 32 + 32
        ts_decode_4x = torch.cat([decode_2x_s, decode_4x], 1)  # 16 + 32
        ts_decode_2x = torch.cat([decode_1x_s, decode_2x], 1)  # 16 + 16
        ts_decode_1x = decode_1x  # 16


        _, _, d, h, w = encode_8x.shape
        fuse_decode_16x = self.lrelu(self.fuse5(ts_decode_16x))
        fuse_decode_16x_up = F.interpolate(fuse_decode_16x, size=(d, h, w), mode='trilinear', align_corners=True)
        fuse_decode_8x_concat = torch.cat([fuse_decode_16x_up, ts_decode_8x], 1)  # 32 + 32 + 32

        _, _, d, h, w = encode_4x.shape
        fuse_decode_8x = self.lrelu(self.fuse4(fuse_decode_8x_concat))  # 32
        fuse_decode_8x_up = F.interpolate(fuse_decode_8x, size=(d, h, w), mode='trilinear', align_corners=True)
        fuse_decode_4x_concat = torch.cat([fuse_decode_8x_up, ts_decode_4x], 1)  # 32 + 16 + 32

        _, _, d, h, w = encode_2x.shape
        fuse_decode_4x = self.lrelu(self.fuse3(fuse_decode_4x_concat))  # 32
        fuse_decode_4x_up = F.interpolate(fuse_decode_4x, size=(d, h, w), mode='trilinear', align_corners=True)
        fuse_decode_2x_concat = torch.cat([fuse_decode_4x_up, ts_decode_2x], 1)  # 32 + 16 + 16

        _, _, d, h, w = concatImgs.shape
        fuse_decode_2x = self.lrelu(self.fuse2(fuse_decode_2x_concat)) # 32
        fuse_decode_2x_up = F.interpolate(fuse_decode_2x, size=(d, h, w), mode='trilinear', align_corners=True)
        fuse_decode_1x_concat = torch.cat([fuse_decode_2x_up, ts_decode_1x], 1)  # 32 + 16

        fuse_decode_1x = self.lrelu(self.fuse1(fuse_decode_1x_concat))  # 32

        flow_fuse = self.flow_f(fuse_decode_1x)

        #############################Fuse############################


        return {#'flow_t': flow * self.flow_multiplier, 'flow_s': flow_s * self.flow_multiplier,
                'flow_f': flow_fuse * self.flow_multiplier,
                'attn_1x_s': torch.sum(torch.mul(fuse_decode_2x, fuse_decode_2x), 1),
                'attn_2x_s': torch.sum(torch.mul(fuse_decode_4x, fuse_decode_4x), 1),
                'attn_4x_s': torch.sum(torch.mul(fuse_decode_8x, fuse_decode_8x), 1),
                'attn_8x_s': torch.sum(torch.mul(fuse_decode_16x, fuse_decode_16x), 1),
                }












