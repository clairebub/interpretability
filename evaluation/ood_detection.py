# Modified from https://github.com/yenchanghsu/out-of-distribution-detection/blob/master/methods/ood_detection.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn import metrics
from metrics.ood import tnr_at_tpr95, detection, AverageMeter
from utils.misc import cov


class Baseline(object):
    def __init__(self, args, cnn, train_loader, val_loader, num_classes):
        self.config = args
        self.num_classes = num_classes

        self.cnn = cnn
        self.cnn.cuda()
        self.cnn.zero_grad()

        self.prepare(train_loader, val_loader)
        self.in_domain_scores = self.get_scores(val_loader)

    def prepare(self, train_loader, val_loader):
        return

    def scoring(self, x):
        # max softmax
        if isinstance(x, dict):
            logits = x['logits']
        else:
            logits = x
        prob = F.softmax(logits, dim=1)
        score, _ = prob.max(dim=1)
        return score

    def get_scores(self, dataloader):
        self.cnn.eval()
        scores = []
        for input, target in dataloader:

            input = input.cuda()
            target = target.cuda()

            if 'cosine' in self.config.model_type:
                # get features before last layer for Mahalanobis methods
                _, _, output = self.cnn.forward(input)
            else:
                output = self.cnn.forward(input)

            score = self.scoring(output)

            scores.extend(score.cpu().detach().numpy())

        return scores

    def ood_eval(self, ood_loader):
        score_in = self.in_domain_scores
        score_out = self.get_scores(ood_loader)

        score_all = np.concatenate([score_in, score_out])
        domain_labels = np.zeros(len(score_all))
        domain_labels[:len(score_in)] = 1

        score_in = np.array(score_in)
        score_out = np.array(score_out)
        score_all = np.array(score_all)

        tnr = tnr_at_tpr95(score_in, score_out)
        detection_error, _ = detection(score_in, score_out)
        auroc = metrics.roc_auc_score(domain_labels, score_all)
        aupr_in = metrics.average_precision_score(domain_labels, score_all)
        aupr_out = metrics.average_precision_score(-1 * domain_labels + 1, 1 - score_all)

        return {'tnr(\u2191)': tnr,
                'det_err(\u2193)': detection_error,
                'auroc(\u2191)': auroc,
                'aupr_in(\u2191)': aupr_in,
                'aupr_out(\u2191)': aupr_out,
                'score_avg(\u2191)': '{:.2e}'.format(score_out.mean()),
                'score_std(\u2193)': '{:.2e}'.format(score_out.std())
                }


class InputPreProcess(Baseline):
    def prepare(self, train_loader, val_loader):
        self.perturb_magnitude = self.search_perturb_magnitude(val_loader)
        print('Inputs are perturbed with magnitude', self.perturb_magnitude)

    def search_perturb_magnitude(self, dataloader):
        if len(self.config.data_perturb_magnitude) == 1:
            return self.config.data_perturb_magnitude[0]
        else:
            magnitude_list = self.config.data_perturb_magnitude
            print('Searching the best perturbation magnitude on in-domain data. Magnitude:', magnitude_list)
            self.cnn.eval()
            loss_list = {}
            for m in magnitude_list:
                loss_meter = AverageMeter()
                for input, _, _ in dataloader:  # Here we don't need labels

                    input = input.cuda().requires_grad_(True)

                    output = self.cnn.forward(input)
                    loss = -self.scoring(output).mean()
                    loss.backward()

                    gradient = torch.ge(input.grad.data, 0)
                    gradient = (gradient.float() - 0.5) * 2
                    modified_input = torch.add(input.detach(), -m, gradient)

                    output = self.cnn.forward(modified_input)
                    loss = -self.scoring(output)
                    loss_meter.update(loss.mean(), len(loss))
                loss_list[m] = loss_meter.avg
                print('Magnitude:', m, 'loss:', loss_list[m])
            best_m = min(loss_list, key=(lambda key: loss_list[key]))
            return best_m

    def get_scores(self, dataloader):
        self.cnn.eval()
        confidence = []
        for input, _ in dataloader:
            input = input.cuda().requires_grad_(True)

            if 'cosine' in self.config.model_type:
                _, _, output = self.cnn.forward(input)
            else:
                output = self.cnn.forward(input)

            loss = -self.scoring(output).mean()
            loss.backward()

            gradient = torch.ge(input.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            modified_input = torch.add(input.detach(), -self.perturb_magnitude, gradient)

            if 'cosine' in self.config.model_type:
                _, _, output = self.cnn.forward(modified_input)
            else:
                output = self.cnn.forward(modified_input)

            score = self.scoring(output)

            confidence.extend(score.cpu().detach().numpy())

        return confidence


class ODIN(InputPreProcess):
    def get_scores(self, dataloader):
        self.cnn.eval()
        scores = []
        for input, target in dataloader:

            input = input.cuda()
            target = target.cuda()

            if 'cosine' in self.config.model_type:
                output, _, _ = self.cnn.forward(input)
            else:
                output = self.cnn.forward(input)

            score = self.scoring(output)

            scores.extend(score.cpu().detach().numpy())

        return scores

    # Temperature scaling + Input preprocess
    def scoring(self, x):
        # max softmax
        if isinstance(x, dict):
            logits = x['logits']
        else:
            logits = x
        logits /= 1000  # Temperature=1000 as suggested in ODIN paper
        prob = F.softmax(logits, dim=1)
        score, _ = prob.max(dim=1)
        return score


class Mahalanobis(Baseline):
    def prepare(self, train_loader, val_loader):
        self.init_mahalanobis(train_loader)

    def init_mahalanobis(self, dataloader):
        if 'cosine' not in self.config.model_type:
            self.cnn.module.fc = torch.nn.Identity()  # So we extract the features

        print('Init: Calculating Mahalanobis ...', len(dataloader))
        all_feat = []
        all_label = []
        for input, target in dataloader:

            input = input.cuda()
            target = target.cuda()

            if 'cosine' in self.config.model_type:
                _, _, feat = self.cnn.forward(input)
            else:
                feat = self.cnn.forward(input)

            all_feat.extend(feat.cpu().detach().numpy())
            all_label.extend(target.cpu().detach().numpy())

        all_feat = torch.from_numpy(np.array(all_feat))
        all_label = torch.from_numpy(np.array(all_label))
        assert all_feat.ndimension() == 2

        all_feat = all_feat.cuda()
        all_label = all_label.cuda()

        self.centers = torch.zeros(self.num_classes, all_feat.size(1), device=all_feat.device)

        for i in range(self.num_classes):
            self.centers[i] = all_feat[all_label == i].mean(dim=0)

        X = all_feat - torch.index_select(self.centers, dim=0, index=all_label)

        # self.precision = X.var(dim=0).pow(-1).diagflat()  # This simplification will cause a significant performance drop
        self.precision = cov(X).pinverse()

    def scoring(self, x):
        diff = x.unsqueeze(dim=1) - self.centers.unsqueeze(dim=0)  # Broadcasting operation

        for i in range(self.num_classes):
            zero_f = diff[:, i]
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, self.precision), zero_f.t()).diag()

            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

        score, _ = gaussian_score.max(dim=1)

        return score


class Mahalanobis_IPP(Mahalanobis, InputPreProcess):
    def prepare(self, train_loader, val_loader):
        self.init_mahalanobis(train_loader)
        self.perturb_magnitude = self.search_perturb_magnitude(val_loader)
        print('Inputs are perturbed with magnitude', self.perturb_magnitude)


class DeepMahalanobis(Baseline):
    def prepare(self, train_loader, val_loader):
        self.init_mahalanobis(train_loader)

    def init_mahalanobis(self, dataloader):

        def new_forward(self, x):
            return self.features(x)
            # return (self.layer1(x), self.layer2(x), self.layer3(x), self.layer4(x))

        self.cnn.module.__class__.forward = new_forward

        print('Init: Calculating DeepMahalanobis ...', len(dataloader))
        input, _ = dataloader.dataset[0]
        input = input.unsqueeze(0).cuda()

        feats = self.cnn.forward(input)

        self.num_out = len(feats)

        all_feat = {i: [] for i in range(self.num_out)}
        self.centers = {i: [] for i in range(self.num_out)}
        self.precision = {i: [] for i in range(self.num_out)}
        all_label = []
        for input, target in dataloader:

            input = input.cuda()
            target = target.cuda()

            feats = self.cnn.forward(input)
            for i in range(self.num_out):
                all_feat[i].extend(feats[i].mean(-1).mean(-1).cpu().detach().numpy())
            all_label.extend(target.cpu().detach().numpy())

        for i in range(self.num_out):
            all_feat[i] = torch.from_numpy(np.array(all_feat[i]))
            all_feat[i] = all_feat[i].cuda()
        all_label = torch.from_numpy(np.array(all_label))
        all_label = all_label.cuda()

        for i in range(self.num_out):
            # feats = torch.cat(all_feat[i])
            feats = all_feat[i]
            assert feats.ndimension() == 2
            self.centers[i] = torch.zeros(self.num_classes, feats.size(1), device=feats.device)
            for c in range(self.num_classes):
                self.centers[i][c] = feats[all_label == c].mean(dim=0)
            X = feats - torch.index_select(self.centers[i], dim=0, index=all_label)
            self.precision[i] = cov(X).pinverse()

    def get_scores(self, dataloader):
        self.cnn.eval()
        scores = []
        for input, target in dataloader:

            input = input.cuda()
            target = target.cuda()

            output = self.cnn.forward(input)

            score = self.scoring(output)

            scores.extend(score.cpu().detach().numpy())

        return scores

    def scoring(self, x):
        deep_scores = torch.zeros(x[0].size(0), self.num_out, device=x[0].device)
        for i in range(self.num_out):
            feat = x[i].mean(-1).mean(-1)
            diff = feat.unsqueeze(dim=1) - self.centers[i].unsqueeze(dim=0)  # Broadcasting operation
            for c in range(self.num_classes):
                zero_f = diff[:, c]
                term_gau = -0.5 * torch.mm(torch.mm(zero_f, self.precision[i]), zero_f.t()).diag()
                if c == 0:
                    gaussian_score = term_gau.view(-1, 1)
                else:
                    gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)
            deep_scores[:, i], _ = gaussian_score.max(dim=1)

        return deep_scores.sum(dim=1)


class DeepMahalanobis_IPP(DeepMahalanobis, InputPreProcess):
    def prepare(self, train_loader, val_loader):
        self.init_mahalanobis(train_loader)
        self.perturb_magnitude = self.search_perturb_magnitude(val_loader)
        print('Inputs are perturbed with magnitude', self.perturb_magnitude)


class Disentangle_SYD(Baseline):
    def scoring(self, x):
        assert isinstance(x, dict), 'The model doesnt provide disentangled results'
        return x['S_YD']


class Disentangle_SYD_IPP(InputPreProcess):
    def scoring(self, x):
        assert isinstance(x, dict), 'The model doesnt provide disentangled results'
        return x['S_YD']


class Disentangle_SD(Baseline):
    def scoring(self, x):
        assert isinstance(x, dict), 'The model doesnt provide disentangled results'
        return x['S_D']


class Disentangle_SD_IPP(InputPreProcess):
    def scoring(self, x):
        assert isinstance(x, dict), 'The model doesnt provide disentangled results'
        return x['S_D']


class OfflineDisentangle_SYD(Baseline):
    def __init__(self, baseObject):
        super(OfflineDisentangle_SYD, self).__init__(baseObject)

    def ood_prepare(self, dataloader):

        self.log('Offline disentangle ...')
        all_out = []
        all_label = []
        for input, target, _ in dataloader:
            # The task id is ignored here

            input = input.cuda()
            target = target.cuda()

            out = self.forward(input)
            all_out.append(out)
            all_label.append(target)
        all_out = torch.cat(all_out)
        all_label = torch.cat(all_label)
        self.num_classes = len(torch.unique(all_label))
        self.std = torch.zeros(self.num_classes, device=input.device)
        for c in range(self.num_classes):
            self.std[c] = all_out[all_label == c].std()
        print('Distance std:', self.std.mean())

    def score(self, x):
        S_YD = x / self.std.view(1, -1)
        return super(Disentangle, self).score(S_YD)
