"""
@author:Jun Wang
@date: 20201123
@contact: jun21wangustc@gmail.com
"""

import math
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class ArcFace(Module):
    """Implementation for "ArcFace: Quality Adaptive Margin for Face Recognition"
    """
    def __init__(self,
                 feat_dim=512, # feat_dim
                 num_class=70722, # num_class
                 m=0.4,
                 h=0.333, # h
                 scale=64., # scale
                 t_alpha=1.0,
                 ):
        super(AdaFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(feat_dim,classnum))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m
        self.eps = 1e-3
        self.h = h
        self.scale = scale

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

    def forward(self, feats, label):
        norms = torch.norm(feats, 2, 1, True)
        features = torch.div(feats, norm)

        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(features, kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.scale
        return scaled_cosine_m