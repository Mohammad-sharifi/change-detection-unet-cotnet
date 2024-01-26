import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.initializers import he_normalizer
from .layers.layers import ConvBlock, BackBone, UpsamplingBlock, EncoderSpaChAttBlock, EncoderBaseBlock


# class SpatialAttentionBlock(nn.Module):
#     def __init__(self, conv_output_dims, reduction_factor=8):
#         super(SpatialAttentionBlock, self).__init__()
#         self.c1 = conv_output_dims[1] // reduction_factor
#         self.c = conv_output_dims[3]
#         self.d = conv_output_dims[1]
#         self.key_embed = self.conv1x1(d, c1)
#         self.query_embed = self.conv1x1(d, c1)
#         self.val_embed = self.conv1x1(d, d)

#     @staticmethod
#     def conv1x1(in_channels, out_channels):
#         net = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1),
#             nn.ReLU()
#         )
#         net[0].weight = net[0].apply(he_normalizer)

#         return net

#     def forward(self, x):
#         """
#         Accepts a 4-D tensor as input (B, C, H, W)
#         """
#         key_embed = self.key_embed(x)
#         key_embed = key_emmbed.view(-1, self.c1, self.c**2)
#         query_embed = self.query_embed(x)
#         query_embed = query_embed.view(-1, self.c1, self.c**2)
#         val_embed = self.val_embed(x)
#         pass


class ChannelAttentionBlock(nn.Module):
    def forward(self, x):
        c = x.size(3)
        d = x.size(1)

        fch = x.view(-1, c*c, d)
        out = torch.linalg.matmul(fch, fch.transpose(1,2))
        out = torch.nn.functional.softmax(out, dim=-1)
        out = torch.linalg.matmul(fch.transpose(1,2), out)
        out = out.view(-1, d, c, c)
        return out


class UnetCotnetNetwork(nn.Module):
    def __init__(self, in_channels, classes):
        super(UnetCotnetNetwork, self).__init__()

        # self.batch_size = batch_size

        self.channel_att = ChannelAttentionBlock()

        # left side blocks
        self.left_backbone = BackBone(in_channels)

        # right side blocks
        self.right_backbone = BackBone(in_channels)

        # upsampling block
        self.upsamples = nn.ModuleDict({
            "512_512": UpsamplingBlock(512, 512, (512, 512), (512, 512)), 
            "256_256": UpsamplingBlock(512, 256, (256, 256), (256, 256)), 
            "128_128": UpsamplingBlock(256, 128, (128, 128), (128, 128)), 
            "128_64": UpsamplingBlock(128, 128, (128, 64), (64, 64), merge_upsample=False), 
        })

        self.conv_bn = nn.Sequential(
            nn.Conv2d(64, classes, 1),
            nn.BatchNorm2d(classes),
            # nn.Sigmoid(), # since we are using BCEWithLogitsLoss
        )

    def forward(self, left_x, right_x):
        # Left part
        left_bb_outputs = self.left_backbone(left_x)

        # Right part
        right_bb_outputs = self.right_backbone(right_x)

        # Channel part
        dist = (left_bb_outputs[-1] - right_bb_outputs[-1]).pow(2)
        dist = self.channel_att(dist)

        # for i in range(len(left_bb_outputs)):
        #     print(f"idx: {i-4}\tconv{2**(i+6)} --> {left_bb_outputs[i].shape}")

        # for i in range(len(right_bb_outputs)):
        #     print(f"idx: {i-4}\tconv{2**(i+6)} --> {right_bb_outputs[i].shape}")

        # print(f"dist.shape --> {dist.shape}\n====================\n")

        # Upsampling part
        up6_out = self.upsamples["512_512"](dist, left_bb_outputs[-2], right_bb_outputs[-2])
        # print("up6_out shape --> ", up6_out.shape)
        up7_out = self.upsamples["256_256"](up6_out, left_bb_outputs[-3], right_bb_outputs[-3])
        # print("up7_out shape --> ", up7_out.shape)
        up8_out = self.upsamples["128_128"](up7_out, left_bb_outputs[-4], right_bb_outputs[-4])
        # print("up8_out shape --> ", up8_out.shape)
        up9_out = self.upsamples["128_64"](up8_out)

        out = self.conv_bn(up9_out)
        return out



# This network is per our talk to combine encoder in a unet style
class UnetAttentionBasedNetwork(nn.Module):
    def __init__(self, in_channels=3, classes=2):
        super(UnetAttentionBasedNetwork, self).__init__()

        # Left side
        self.intro_ll1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.MaxPool2d(2,2)
        )
        self.intro_ll2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.MaxPool2d(2,2)
        )
        self.intro_ll3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.MaxPool2d(2,2)
        )
        self.intro_ll4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.MaxPool2d(2,2)
        )

        self.encoder_ll1 = EncoderBaseBlock(in_channels=64, num_heads=2)
        self.encoder_ll2 = EncoderBaseBlock(in_channels=128, num_heads=4)
        self.encoder_ll3 = EncoderSpaChAttBlock(in_channels=256, num_heads=8)
        self.encoder_ll4 = EncoderSpaChAttBlock(in_channels=512, num_heads=16)

        # Right side
        self.intro_rl1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.MaxPool2d(2,2)
        )
        self.intro_rl2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.MaxPool2d(2,2)
        )
        self.intro_rl3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.MaxPool2d(2,2)
        )
        self.intro_rl4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.MaxPool2d(2,2)
        )

        self.encoder_rl1 = EncoderBaseBlock(in_channels=64, num_heads=2)
        self.encoder_rl2 = EncoderBaseBlock(in_channels=128, num_heads=4)
        self.encoder_rl3 = EncoderSpaChAttBlock(in_channels=256, num_heads=8)
        self.encoder_rl4 = EncoderSpaChAttBlock(in_channels=512, num_heads=16)

        # Decoder side
        self.upsamples = nn.ModuleDict({
            "512_512": UpsamplingBlock(512, 512, (512, 512), (512, 512)), 
            "256_256": UpsamplingBlock(512, 256, (256, 256), (256, 256)), 
            "128_128": UpsamplingBlock(256, 128, (128, 128), (128, 128)), 
            "128_64": UpsamplingBlock(128, 128, (128, 64), (64, 64), merge_upsample=False), 
        })

        self.conv_bn = nn.Sequential(
            nn.Conv2d(64, classes, 1),
            nn.BatchNorm2d(classes),
        )

    def forward(self, x_left, x_right):
        # print("starting shape is: ")
        # print(x_left.shape, x_right.shape)
        x_left1 = self.encoder_ll1(self.intro_ll1(x_left))
        x_left2 = self.encoder_ll2(self.intro_ll2(x_left1))
        x_left3 = self.encoder_ll3(self.intro_ll3(x_left2))
        x_left4 = self.encoder_ll4(self.intro_ll4(x_left3))

        x_right1 = self.encoder_rl1(self.intro_rl1(x_right))
        x_right2 = self.encoder_rl2(self.intro_rl2(x_right1))
        x_right3 = self.encoder_rl3(self.intro_rl3(x_right2))
        x_right4 = self.encoder_rl4(self.intro_rl4(x_right3))

        # print("----------------")
        # print(x_left1.shape, x_right1.shape)
        # print(x_left2.shape, x_right2.shape)
        # print(x_left3.shape, x_right3.shape)
        # print(x_left4.shape, x_right4.shape)

        dist = (x_left4 - x_right4).pow(2)

        up6_out = self.upsamples["512_512"](dist, x_left3, x_right3)
        # print("up6_out shape --> ", up6_out.shape)
        up7_out = self.upsamples["256_256"](up6_out, x_left2, x_right2)
        # print("up7_out shape --> ", up7_out.shape)
        up8_out = self.upsamples["128_128"](up7_out, x_left1, x_right1)
        # print("up8_out shape --> ", up8_out.shape)
        up9_out = self.upsamples["128_64"](up8_out)
        # print("up9_out shape --> ", up9_out.shape)

        return self.conv_bn(up9_out)

