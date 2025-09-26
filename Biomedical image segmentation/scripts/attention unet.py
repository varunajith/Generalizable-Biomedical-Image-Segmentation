# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 19:18:10 2025

@author: Sayantari
"""
from einops import rearrange
from torch import einsum
import torch.nn as nn
import functools
import torch
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Attention Gate Module.
        Args:
            F_g (int): Number of channels in the decoder feature map.
            F_l (int): Number of channels in the encoder feature map.
            F_int (int): Intermediate number of channels.
        """
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Forward pass for the Attention Gate.
        Args:
            g (Tensor): Decoder feature map (lower resolution).
            x (Tensor): Encoder feature map (higher resolution).
        Returns:
            Tensor: Attention-weighted encoder feature map.
        """
        g1 = self.W_g(g)  # Transform decoder features
        x1 = self.W_x(x)  # Transform encoder features
        psi = self.relu(g1 + x1)  # Add and apply ReLU
        psi = self.psi(psi)  # Sigmoid to get attention coefficients
        #Temp = 0.4
        #psi = self.psi(psi/Temp)
        #output = torch.sigmoid(x*psi+x)
        
        #print(f"Attention coefficients min: {psi.min().item()}, max: {psi.max().item()}")
        
        #return output, psi
        return x*psi, psi # Apply attention coefficients to encoder features

class encoder_attention_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder_attention_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        s = self.conv(x)  # Convolutional block
        p = self.pool(s)  # Max pooling
        return s, p

class decoder_attention_block(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=True):
        super(decoder_attention_block, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Add dropout
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.use_attention = use_attention
        if use_attention:
            self.attention_gate = AttentionGate(F_g=out_channels, F_l=out_channels, F_int=out_channels // 2)

    def forward(self, x, skip):
        x = self.up(x)  # Upsample
        if x.size()[2:] != skip.size()[2:]:  # Ensure spatial dimensions match
            x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=True)
        if self.use_attention:
            skip,psi = self.attention_gate(x, skip)  # Apply attention gate
        x = torch.cat([x, skip], dim=1)  # Concatenate with skip connection
        x = self.conv(x)  # Convolutional block
        return x,psi
class AttentionUnetSegmentation(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, use_attention=True):
        super(AttentionUnetSegmentation, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_attention = use_attention

        """ Encoder """
        self.e1 = encoder_attention_block(n_channels, 64)
        self.e2 = encoder_attention_block(64, 128)
        self.e3 = encoder_attention_block(128, 256)
        self.e4 = encoder_attention_block(256, 512)

        """ Bottleneck """
        
        self.b = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        """ Decoder """
        self.d1 = decoder_attention_block(1024, 512, use_attention=use_attention)
        self.d2 = decoder_attention_block(512, 256, use_attention=use_attention)
        self.d3 = decoder_attention_block(256, 128, use_attention=use_attention)
        self.d4 = decoder_attention_block(128, 64, use_attention=use_attention)

        """ Classifier """
        self.outputs = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1,psi1 = self.d1(b, s4)
        d2,psi2 = self.d2(d1, s3)
        d3,psi3 = self.d3(d2, s2)
        d4,psi4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(d4)
        return outputs,psi4