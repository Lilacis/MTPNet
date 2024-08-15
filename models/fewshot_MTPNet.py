"""
FSS via MTPNet
Extended from ADNet code by Hansen et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.encoder import Res101Encoder
import numpy as np
import random
import cv2
from models.moudles import MLP, Decoder

class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()

        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)  # or "resnet101"
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.threshold = 0.5
        self.criterion = nn.NLLLoss()
        self.criterion_MSE = nn.MSELoss()
        self.cons_num = 1000  # ensure adequate structural points
        self.topk_num = 5  # a
        self.sample_num = 100   # 2*N_sml
        self.mlp = MLP(self.cons_num, self.sample_num)
        self.decoder = Decoder(self.sample_num)
        self.mse_loss_fn = nn.MSELoss()

    def forward(self, supp_imgs, supp_mask, qry_imgs, train=False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """

        self.n_ways = len(supp_imgs)  # 1
        self.n_shots = len(supp_imgs[0])  # 1
        self.n_queries = len(qry_imgs)  # 1
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]    # 1
        supp_bs = supp_imgs[0][0].shape[0]      # 1
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)      # (2, 3, 256,256)
        # encoder output
        img_fts, tao = self.encoder(imgs_concat)
        supp_fts = [img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view(  # B x Wa x Sh x C x H' x W'
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        supp_fts = supp_fts[0]

        qry_fts = [img_fts[dic][self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'
            qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        qry_fts = qry_fts[0]

        ##### Get threshold #######
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]
        self.thresh_pred = [self.t for _ in range(self.n_ways)]

        ###### Compute loss ######
        align_loss = torch.zeros(1).to(self.device)
        mse_loss = torch.zeros(1).to(self.device)
        outputs = []
        for epi in range(supp_bs):

            ###### Extract prototypes ######
            if supp_mask[epi][0].sum() == 0:
                supp_fts_ = [[self.getFeatures(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot])
                               for shot in range(self.n_shots)] for way in range(self.n_ways)]
                fg_prototypes = self.getPrototype(supp_fts_)

                ###### Get query predictions ######
                qry_pred = torch.stack(
                    [self.getPred(qry_fts[epi], fg_prototypes[way], self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1)  # 2 x N x Wa x H' x W'
                preds = F.interpolate(qry_pred, size=img_size, mode='bilinear', align_corners=True)
                preds = torch.cat((1.0 - preds, preds), dim=1)
                outputs.append(preds)
                if train:
                    align_loss_epi = self.alignLoss([supp_fts[epi]], [qry_fts[epi]], preds, supp_mask[epi])
                    align_loss += align_loss_epi
            else:
                fg_pts = [[self.get_fg_pts(supp_fts[[epi], way, shot], qry_fts[way], supp_mask[[epi], way, shot])
                           for shot in range(self.n_shots)] for way in range(self.n_ways)]
                fg_pts = self.get_all_prototypes(fg_pts)
                bg_pts = [[self.get_bg_pts(supp_fts[[epi], way, shot], qry_fts[way], supp_mask[[epi], way, shot])
                           for shot in range(self.n_shots)] for way in range(self.n_ways)]
                bg_pts = self.get_all_prototypes(bg_pts)

                ###### Get query predictions ######
                fg_sim = torch.stack(
                    [self.get_sim(qry_fts[epi], fg_pts[way]) for way in range(self.n_ways)], dim=1).squeeze(0)
                bg_sim = torch.stack(
                    [self.get_sim(qry_fts[epi], bg_pts[way]) for way in range(self.n_ways)], dim=1).squeeze(0)
                
                preds_fg = F.interpolate(fg_sim, size=img_size, mode='bilinear', align_corners=True)
                preds_bg = F.interpolate(bg_sim, size=img_size, mode='bilinear', align_corners=True)

                preds = torch.cat([preds_fg, preds_bg], dim=1)
                preds = torch.softmax(preds, dim=1)
    
                outputs.append(preds)
                if train:
                    align_loss_epi = self.align_aux_Loss([supp_fts[epi]], [qry_fts[epi]], preds,
                                                         supp_mask[epi])  # fg_pts, bg_pts
                    mse_loss_epi = self.get_mse_loss(supp_fts[epi], supp_mask[epi], fg_pts, bg_pts)

                    align_loss += align_loss_epi
                    mse_loss += mse_loss_epi


        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])

        return output, align_loss / supp_bs, mse_loss / supp_bs

    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes
        # (1, 512, 64, 64) (1, 512), (1, 1)
        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))   # ([1, 64, 64])

        return pred

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C] (1, 1, (1, 512))
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  # concat all fg_fts   (n_way, (1, 512))


        return fg_prototypes

    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [self.getFeatures(qry_fts[0], pred_mask[way + 1])]
                fg_prototypes = self.getPrototype([qry_fts_])

                # Get predictions
                supp_pred = self.getPred(supp_fts[0][way, [shot]], fg_prototypes[way], self.thresh_pred[way])  # N x Wa x H' x W'
                supp_pred = F.interpolate(supp_pred[None, ...], size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)

                # Combine predictions of different feature maps
                pred_ups = torch.cat((1.0 - supp_pred, supp_pred), dim=1)  # (1, 2, 256, 256)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss

    def align_aux_Loss(self, supp_fts, qry_fts, pred, fore_mask):
        """
            supp_fts: [1, 512, 64, 64]
            qry_fts: (1, 512, 64, 64)
            pred: [1, 2, 256, 256]
            fore_mask: [Way, Shot , 256, 256]

        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes         1 x C x H' x W'   1 x H x W
                fg_pts = [self.get_fg_pts(qry_fts[0], supp_fts[0], pred_mask[way + 1])]
                fg_prototypes = self.get_all_prototypes([fg_pts])
                bg_pts = [self.get_bg_pts(qry_fts[0], supp_fts[0], pred_mask[way + 1])]
                bg_prototypes = self.get_all_prototypes([bg_pts])

                # Get predictions
                supp_pred_fg = self.get_sim(supp_fts[0][way, [shot]], fg_prototypes[way])   # N x Wa x H' x W'
                supp_pred_bg = self.get_sim(supp_fts[0][way, [shot]], bg_prototypes[way])   # N x Wa x H' x W'
                
                pred_fg = F.interpolate(supp_pred_fg, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                pred_bg = F.interpolate(supp_pred_bg, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                supp_pred = torch.cat([pred_fg, pred_bg], dim=1)
                preds = torch.softmax(supp_pred, dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(preds, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss

    def get_fg_pts(self, features, qry_features, mask):
        """
        feature: 输入tensor 1 x C x H x W
        mask: 输出tensor 1 x H x W
        prototypes: 输出tensor  2*N_sml x C
        """
        mask = F.interpolate(mask.unsqueeze(dim=0), size=features.shape[-2:], mode='bilinear',
                             align_corners=True).squeeze(dim=0)  # [1, 64, 64]
        # threshold processing
        mask = (mask >= self.threshold).float()
        
        fg_fts, mask = self.sparse_supp_sampling(features, mask)
        fg_prototypes = self.cross_layers(qry_features, fg_fts, mask)
        exf_prototypes = self.extend_prototypes(fg_prototypes, fg_fts).to(self.device)
        exf_prototypes = self.mlp(exf_prototypes).permute(1, 0)
        
        return exf_prototypes

    def get_bg_pts(self, features, qry_features, mask):
        """
        feature: 输入tensor 1 x C x H x W
        mask: 输出tensor 1 x H x W
        prototypes: 输出tensor  2*N_sml x C
        """
        mask = 1 - mask
        mask = F.interpolate(mask.unsqueeze(dim=0), size=features.shape[-2:], mode='bilinear',
                             align_corners=True).squeeze(dim=0)  # [1, 64, 64]
        # threshold processing
        mask = (mask >= self.threshold).float()

        bg_fts, mask = self.sparse_supp_sampling(features, mask)
        bg_prototypes = self.cross_layers(qry_features, bg_fts, mask)
        exb_prototypes = self.extend_prototypes(bg_prototypes, bg_fts).to(self.device)
        exb_prototypes = self.mlp(exb_prototypes).permute(1, 0)
        
        return exb_prototypes

    def cross_layers(self, qry_fts, supp_fts, supp_mask):
        qry_fts = qry_fts.view(-1, qry_fts.shape[-1] * qry_fts.shape[-2], qry_fts.shape[-3])
        supp_mask = supp_mask.view(-1)  # [2*cons_num]

        B, N, C = qry_fts.shape
        N_s = supp_fts.size(0)

        q = qry_fts.reshape(B, N, -1, C).permute(0, 2, 1, 3)
        k = supp_fts.reshape(B, N_s, -1, C).permute(0, 2, 1, 3)

        # [bs, nH, n, ns]->similarity
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1)  # [H*W, 2*cons_num]

        ######### use average top a similarist points ###########
        attn_single_hd = attn[0, 0, ...]
        k2q_sim_idx = attn_single_hd.topk(self.topk_num, dim=0)[1]  # [a, 2*cons_num]
        flatten_indices = k2q_sim_idx.flatten()
        qry_fts_ = qry_fts[:, flatten_indices, :]
        qry_fts_ = qry_fts_.reshape(N_s, self.topk_num, C)
        qry_fts_mean = qry_fts_.mean(dim=1)  # [2*cons_num, c]

        ######### new similarity map of mean q and k #######
        N_ = qry_fts_mean.size(0)
        q_ = qry_fts_mean.reshape(B, N_, -1, C).permute(0, 2, 1, 3)
        q_ = F.normalize(q_, dim=-1)
        attn_ = q_ @ k.transpose(-2, -1)
        attn_single_hd_ = attn_[0, 0, ...]
        q2k_sim_idx = attn_single_hd_.max(1)[1]

        # True means matched position in supp
        re_map_mask = torch.gather(supp_mask, 0, q2k_sim_idx)
        asso_single_head = (re_map_mask == 1) & (supp_mask == 1).to(self.device)
        qry_fts = qry_fts_mean[asso_single_head, :]  # [Num(<cons_num), c]

        return qry_fts

    def extend_prototypes(self, prototypes, fg_sml):
        num_pros = len(prototypes)
        prototypes = prototypes.permute(1, 0)
        fg_sml = fg_sml.permute(1, 0)  # [c, 2*cons_num]
        N = self.cons_num

        # align pros by query points
        if num_pros == 0:
            target_shape = [512, N]
            ex_prototypes = torch.zeros(target_shape, dtype=prototypes.dtype)
            ex_prototypes[:, :prototypes.shape[1]] = prototypes
        else:
            r = N // num_pros + 1
            ex_prototypes = prototypes.repeat(1, r)[:, :N]

        # extend pros by support points
        sup_prototypes = fg_sml[:, :N]
        aln_prototypes = torch.cat((ex_prototypes.to(self.device), sup_prototypes.to(self.device)), dim=1)

        return aln_prototypes

    def sparse_supp_sampling(self, s_x, supp_mask):
        s_x = s_x.squeeze(0)
        s_x = s_x.permute(1, 2, 0)
        s_x = s_x.view(s_x.shape[-2] * s_x.shape[-3], s_x.shape[-1])
        supp_mask = supp_mask.squeeze(0).view(-1)

        rand_fg_num = self.cons_num
        rand_bg_num = self.cons_num

        num_fg = (supp_mask == 1).sum()
        num_bg = (supp_mask == 0).sum()

        fg_k = s_x[supp_mask == 1]  # [num_fg, c]
        bg_k = s_x[supp_mask == 0]  # [num_bg, c]
        fg_idx = torch.nonzero(supp_mask == 1, as_tuple=False).squeeze(dim=1)  # [num_fg]
        bg_idx = torch.nonzero(supp_mask == 0, as_tuple=False).squeeze(dim=1)  # [num_bg]
        fg_mask = supp_mask[fg_idx]
        bg_mask = supp_mask[bg_idx]

        if num_fg < rand_fg_num:
            if num_fg == 0:
                re_k = torch.zeros(rand_fg_num, 512).to(self.device)
                re_mask = torch.zeros(rand_fg_num).to(self.device)

                k_b = random.sample(range(num_bg), rand_bg_num)
                re_k = torch.cat([re_k, bg_k[k_b]], dim=0)
                re_mask = torch.cat([re_mask, bg_mask[k_b]], dim=0)

            else:
                r = rand_fg_num // num_fg
                k_f = random.sample(range(num_fg), rand_fg_num % num_fg)
                k_b = random.sample(range(num_bg), rand_bg_num)

                re_k = torch.cat([fg_k for _ in range(r)], dim=0)
                re_k = torch.cat([fg_k[k_f], re_k, bg_k[k_b]], dim=0)
                re_mask = torch.cat([fg_mask for _ in range(r)], dim=0)
                re_mask = torch.cat([fg_mask[k_f], re_mask, bg_mask[k_b]], dim=0)

        elif num_bg < rand_bg_num :
            if num_bg == 0:
                re_k = torch.zeros(rand_bg_num, 512).to(self.device)
                re_mask = torch.zeros(rand_bg_num).to(self.device)

                k_f = random.sample(range(num_fg), rand_fg_num)
                re_k = torch.cat([fg_k[k_f], re_k], dim=0)
                re_mask = torch.cat([fg_mask[k_f], re_mask], dim=0)

            else:
                r = rand_bg_num // num_bg
                k_b = random.sample(range(num_bg), rand_bg_num % num_bg)
                k_f = random.sample(range(num_fg), rand_fg_num)

                re_k = torch.cat([bg_k for _ in range(r)], dim=0)
                re_k = torch.cat([fg_k[k_f], bg_k[k_b], re_k], dim=0)
                re_mask = torch.cat([bg_mask for _ in range(r)], dim=0)
                re_mask = torch.cat([fg_mask[k_f], bg_mask[k_b], re_mask], dim=0)

        else:
            k_b = random.sample(range(num_bg), rand_bg_num)
            k_f = random.sample(range(num_fg), rand_fg_num)
            re_k = torch.cat([fg_k[k_f], bg_k[k_b]], dim=0)
            re_mask = torch.cat([fg_mask[k_f], bg_mask[k_b]], dim=0)

        return re_k, re_mask

    def get_all_prototypes(self, fg_fts):
        """
            fg_fts: lists of list of tensor
                        expect shape: Wa x Sh x [all x C]
            fg_prototypes: [(all, 512) * way]    list of tensor
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        prototypes = [sum([shot for shot in way]) / n_shots for way in fg_fts]
        return prototypes

    def get_sim(self, fts, prototypes):
        """
        Calculate the distance between features and prototypes
        # (1, 512, 64, 64) (2*N_sml, 512)
        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        fts_ = fts.permute(0, 2, 3, 1)
        fts_ = F.normalize(fts_, dim=-1)
        pts_ = F.normalize(prototypes, dim=-1)
        fg_sim = torch.matmul(fts_, pts_.transpose(0, 1)).permute(0, 3, 1, 2)
        fg_sim = self.decoder(fg_sim)

        return fg_sim   # [1, 1, 64, 64]

    def get_mse_loss(self, supp_fts, mask, qry_fg_pros, qry_bg_pros):
        n_ways, n_shots = len(mask), len(mask[0])

        supp_fg_pts = [[self.get_fg_pts(supp_fts[[way], shot], supp_fts[[way], shot], mask[[way], shot])
                        for shot in range(self.n_shots)] for way in range(self.n_ways)]
        fg_pts = self.get_all_prototypes(supp_fg_pts)
        supp_bg_pts = [[self.get_bg_pts(supp_fts[[way], shot], supp_fts[[way], shot], mask[[way], shot])
                        for shot in range(self.n_shots)] for way in range(self.n_ways)]
        bg_pts = self.get_all_prototypes(supp_bg_pts)

        loss_mse = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            mse_fg_loss = self.mse_loss_fn(fg_pts[way], qry_fg_pros[way])
            mse_bg_loss = self.mse_loss_fn(bg_pts[way], qry_bg_pros[way])
            loss_mse += mse_fg_loss + mse_bg_loss

        return loss_mse

