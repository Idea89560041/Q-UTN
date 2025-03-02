from torch import nn
import torch
import torch.nn.functional as F
import functools
import numpy as np
from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange
import torchvision.models as models
from torchvision.models import VGG16_Weights


##################################################################################
# Generator
##################################################################################

class SimpleAdaInGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params):
        super(SimpleAdaInGen, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']

        # content encoder
        self.enc = AdaEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)  # "in"
        self.dec = Decoder(n_downsample, n_res, self.enc.output_dim, input_dim, res_norm='adain', activ='sigmoid',
                           pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp_enc = MLP(style_dim, self.get_num_adain_params(self.enc), mlp_dim, 3, norm='none', activ=activ)
        self.mlp_dec = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, images, bvec):
        adain_params_enc = self.mlp_enc(bvec)
        self.assign_adain_params(adain_params_enc, self.enc)
        content = self.enc(images)
        adain_params_dec = self.mlp_dec(bvec)
        self.assign_adain_params(adain_params_dec, self.dec)
        images_recon = self.dec(content)
        return images_recon

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

class ResAdaInGen(SimpleAdaInGen):
    def __init__(self, input_dim, output_dim, params):
        super(SimpleAdaInGen, self).__init__()
        dim = params['dim']
        dim_enc = params['dim_enc']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']
        input_domain = params['input_domain']

        # content encoder
        self.enc = AdaEncoder(input_dim, dim_enc, 'adain', activ, pad_type=pad_type)  # "in"

        self.dec = Decoder(n_downsample, n_res, self.enc.output_dim * input_domain, output_dim, res_norm='adain', activ='none',
                           pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp_enc = MLP(style_dim, self.get_num_adain_params(self.enc), mlp_dim, 3, norm='none', activ=activ)
        self.mlp_dec = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

        self.multi_modal_attention = MultiModalAttentionLayer(input_domain)
        self.dim_reduce_conv = nn.Sequential(
            nn.Conv2d(dim * 4 * input_domain, dim * 4, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, input, bvec):
        adain_params_enc = self.mlp_enc(bvec)
        self.assign_adain_params(adain_params_enc, self.enc)
        b0, t2, t1 = torch.split(input, 1, dim=1)
        content_b0 = self.enc(b0)
        content_t2 = self.enc(t2)
        content_t1 = self.enc(t1)
        content_all = self.multi_modal_attention([content_b0, content_t2, content_t1])
        # content = self.dim_reduce_conv(content_all)
        adain_params_dec = self.mlp_dec(bvec)
        self.assign_adain_params(adain_params_dec, self.dec)
        dwi = self.dec(content_all)
        return dwi

##################################################################################
# Discriminator
##################################################################################
class NLayerProjectionDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator with conditional projection"""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, embed=False, embedding_dim=256, crop_size=96, gradient_penalty=0.):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerProjectionDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.gradient_penalty = gradient_penalty
        kw = 4
        padw = 1
        sequence = [nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
                    nn.LeakyReLU(0.2, True)]
        out_size = self.get_output_size(crop_size, kw, padw, 2)
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                #norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            out_size = self.get_output_size(out_size, kw, padw, 2)

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            #norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        out_size = self.get_output_size(out_size, kw, padw, 1)

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
        out_size = self.get_output_size(out_size, kw, padw, 1)

        if embed:
            self.dim=embedding_dim
            #print(out_size)
            self.embedding = nn.Linear(4, embedding_dim) #nn.Linear(3, int(out_size*out_size*1*embedding_dim))
            self.fc = nn.Conv2d(1, embedding_dim, 3, padding=1)

    def get_output_size(self, in_size, kernel_size, padding, stride):
        return int((in_size - kernel_size + 2*padding)/stride + 1)

    def forward(self, input, y=None):
        # y as a condition for projection Discriminator, if None, return standard forward
        # if y is given, dimension will be batch size * embedding dimension
        """Standard forward."""
        if y is None:
            return self.model(input)
        else:
            out = self.model(input)
            out_emb = self.fc(out)
            bs, c, w, h = out_emb.size()
            y_emb = self.embedding(y).view(-1, self.dim, 1, 1).expand_as(out_emb)
            out = out + torch.sum(y_emb * out_emb, 1, keepdim=True)
            return out


    def get_D_loss(self, real_in, real_img, real_bvec, fake_in, fake_img, fake_bvec, criterionGAN):
        """Calculate GAN loss for the discriminator, from pix2pix"""
        '''
        input: b0
        real_img: target dwi, 
        real_bvec: bvec for real_img
        fake_img: generated image
        fake_bvec: bvec for generated img
        in case we want to use unpaired sampling, or the real_bvec & fake_bvec would be the same
        '''
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat([fake_in, fake_img.detach()], dim=1)
        pred_fake = self.forward(fake_AB, fake_bvec)
        loss_D_fake = criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat([real_in, real_img], dim=1)
        pred_real = self.forward(real_AB, real_bvec)
        loss_D_real = criterionGAN(pred_real, True)

        loss_D_GAN = (loss_D_fake + loss_D_real) * 0.5
        # Calculate gradients of probabilities with respect to examples
        if self.gradient_penalty > 0:
            return loss_D_GAN, pred_real, real_AB

        return loss_D_GAN

    def get_G_loss(self, input, fake_img, fake_emb, criterionGAN):
        """Calculate GAN loss for the generator"""
        # G(A) should fake the discriminator
        fake_AB = torch.cat([input, fake_img], dim=1)
        pred_fake = self.forward(fake_AB, fake_emb)
        loss_G_GAN = criterionGAN(pred_fake, True)
        return loss_G_GAN

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class Unet_Discriminator(nn.Module):

    def __init__(self, input_nc=1, ndf=64, kw=3, padw=1, n_layers=3, n_latent=2, output_nc=1, use_bias=True, embed=True, device='cuda:0'):
        super(Unet_Discriminator, self).__init__()
        self.conditional = embed
        self.head = nn.Sequential(*[Conv2dBlock(input_nc, ndf, stride=1, kernel_size=kw, norm='sn', activation='lrelu', padding=padw),
                    nn.LeakyReLU(0.2, True)])
        nf_mult = 1
        nf_mult_prev = 1
        self.enc_layers = []
        self.n_downsample = n_layers
        for n in range(1, n_layers+1):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.enc_layers.append(nn.Sequential(*[
                Conv2dBlock(ndf * nf_mult_prev, ndf * nf_mult, stride=1, kernel_size=kw, norm='sn', activation='lrelu',
                            pad_type='zero', padding=padw),
                nn.AvgPool2d(2),
                nn.LeakyReLU(0.2, True)
            ]))
        self.enc_layers = nn.ModuleList(self.enc_layers)

        self.latent = []
        for i in range(n_latent):
            self.latent +=[DResBlock(ndf * nf_mult, ndf * nf_mult, kw=kw, padding=padw),
            nn.LeakyReLU(0.2, True)]
        self.latent = nn.Sequential(*self.latent)
        self.linear = nn.Linear(ndf * nf_mult, 1)

        if embed:
            self.embedding_middle = nn.Linear(4, ndf * nf_mult)
            self.embedding_out = nn.Linear(4, output_nc)
            self.fc = nn.Conv2d(1, ndf * nf_mult, 4, padding=1)
        self.dec_layers = []
        for n in range(1, n_layers+1):
            nf_mult_prev = nf_mult
            nf_mult = nf_mult // 2
            self.dec_layers.append(nn.Sequential(*[
                Conv2dBlock(ndf * (nf_mult_prev)* 2, ndf * nf_mult, stride=1, kernel_size=kw, norm='sn',
                            activation='lrelu', pad_type='zero', padding=padw),
                nn.LeakyReLU(0.2, True)
            ]))
        self.dec_layers = nn.ModuleList(self.dec_layers)
        self.last = Conv2dBlock(ndf * nf_mult, 1, stride=1, kernel_size=kw, norm='sn',
                            activation='none', pad_type='zero', padding=padw)

    def init_weight(self):
        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

        return self.apply(init_func)

    def forward(self, x, y=None):
        conditional = False
        if y is not None:
            conditional = True
        h = x
        res_features = []
        h = self.head(h)
        for n in range(self.n_downsample):
            h = self.enc_layers[n](h)
            res_features.append(h)
        h = self.latent(h)
        h_ = h
        h_ = torch.sum(h_, [2,3])
        bottleneck_out = self.linear(h_)

        for n in range(self.n_downsample):
            eid = -1 - n
            h = torch.cat((res_features[eid], h), dim=1)
            h = F.interpolate(h, scale_factor=2)
            h = self.dec_layers[n](h)

        out = self.last(h)
        if conditional:
            emb_mid = self.embedding_middle(y)
            proj_mid = torch.sum(emb_mid * bottleneck_out, 1, keepdim=True)
            bottleneck_out = bottleneck_out + proj_mid

            emb_out = self.embedding_out(y)
            emb_out = emb_out.view(emb_out.size(0), emb_out.size(1), 1, 1).expand_as(out)
            proj = torch.sum(emb_out * out, 1, keepdim=True)
            out = out + proj
        return out, bottleneck_out

    def gan_forward(self, input_real, real_img, real_bvec, input_fake, fake_img, fake_bvec):
        fake_AB = torch.cat((input_fake, fake_img.detach()), dim=1)
        real_AB = torch.cat((input_real, real_img), dim=1)
        self.pred_fake_pix, pred_fake_img = self.forward(fake_AB, fake_bvec)
        self.pred_real_pix, pred_real_img = self.forward(real_AB, real_bvec)


    def get_D_loss(self, input_real, real_img, real_bvec, input_fake, fake_img, fake_bvec, criterionGAN):
        """Calculate GAN loss for the discriminator"""
        fake_AB = torch.cat((input_fake, fake_img.detach()), dim=1)
        pred_fake_pix, pred_fake_img = self.forward(fake_AB, fake_bvec)
        loss_D_fake_pix, loss_D_fake_img = criterionGAN(pred_fake_pix, False), criterionGAN(pred_fake_img, False)
        # Real
        real_AB = torch.cat((input_real, real_img), dim=1)
        # print(real_AB.size())
        pred_real_pix, pred_real_img = self.forward(real_AB, real_bvec)
        loss_D_real_pix, loss_D_real_img = criterionGAN(pred_real_pix, True), criterionGAN(pred_real_img, True)
        # combine loss and calculate gradients
        loss_D_global = (loss_D_fake_img + loss_D_real_img) * 0.5
        loss_D_local = (loss_D_fake_pix + loss_D_real_pix) * 0.5

        return loss_D_global, loss_D_local

    def get_G_loss(self, input, fake_img, fake_emb, criterionGAN):
        """Calculate GAN loss for the generator"""
        # G(A) should fake the discriminator
        fake_AB = torch.cat((input, fake_img), dim=1)
        pred_fake_pix, pred_fake_img = self.forward(fake_AB, fake_emb)
        loss_G_global = criterionGAN(pred_fake_pix, True)
        loss_G_local  = criterionGAN(pred_fake_img, True)
        return loss_G_global, loss_G_local

class VGG16(nn.Module):
  def __init__(self):
    super(VGG16, self).__init__()
    features = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
    self.relu1_1 = torch.nn.Sequential()
    self.relu1_2 = torch.nn.Sequential()

    self.relu2_1 = torch.nn.Sequential()
    self.relu2_2 = torch.nn.Sequential()

    self.relu3_1 = torch.nn.Sequential()
    self.relu3_2 = torch.nn.Sequential()
    self.relu3_3 = torch.nn.Sequential()

    self.relu4_1 = torch.nn.Sequential()
    self.relu4_2 = torch.nn.Sequential()
    self.relu4_3 = torch.nn.Sequential()

    self.relu5_1 = torch.nn.Sequential()
    self.relu5_2 = torch.nn.Sequential()
    self.relu5_3 = torch.nn.Sequential()

    for x in range(2):
      self.relu1_1.add_module(str(x), features[x])

    for x in range(2, 4):
      self.relu1_2.add_module(str(x), features[x])

    for x in range(4, 7):
      self.relu2_1.add_module(str(x), features[x])

    for x in range(7, 9):
      self.relu2_2.add_module(str(x), features[x])

    for x in range(9, 12):
      self.relu3_1.add_module(str(x), features[x])

    for x in range(12, 14):
      self.relu3_2.add_module(str(x), features[x])

    for x in range(14, 16):
      self.relu3_3.add_module(str(x), features[x])

    for x in range(16, 18):
      self.relu4_1.add_module(str(x), features[x])

    for x in range(18, 21):
      self.relu4_2.add_module(str(x), features[x])

    for x in range(21, 23):
      self.relu4_3.add_module(str(x), features[x])

    for x in range(23, 26):
      self.relu5_1.add_module(str(x), features[x])

    for x in range(26, 28):
      self.relu5_2.add_module(str(x), features[x])

    for x in range(28, 30):
      self.relu5_3.add_module(str(x), features[x])

    # don't need the gradients, just want the features
    # for param in self.parameters():
    #    param.requires_grad = False

  def forward(self, x, layers=None, encode_only=False, resize=False):
    x = torch.cat([x, x, x], dim=1)
    relu1_1 = self.relu1_1(x)
    relu1_2 = self.relu1_2(relu1_1)

    relu2_1 = self.relu2_1(relu1_2)
    relu2_2 = self.relu2_2(relu2_1)

    relu3_1 = self.relu3_1(relu2_2)
    relu3_2 = self.relu3_2(relu3_1)
    relu3_3 = self.relu3_3(relu3_2)

    relu4_1 = self.relu4_1(relu3_3)
    relu4_2 = self.relu4_2(relu4_1)
    relu4_3 = self.relu4_3(relu4_2)

    relu5_1 = self.relu5_1(relu4_3)
    relu5_2 = self.relu5_2(relu5_1)
    relu5_3 = self.relu5_3(relu5_2)

    out = {
      'relu1_1': relu1_1,
      'relu1_2': relu1_2,

      'relu2_1': relu2_1,
      'relu2_2': relu2_2,

      'relu3_1': relu3_1,
      'relu3_2': relu3_2,
      'relu3_3': relu3_3,

      'relu4_1': relu4_1,
      'relu4_2': relu4_2,
      'relu4_3': relu4_3,

      'relu5_1': relu5_1,
      'relu5_2': relu5_2,
      'relu5_3': relu5_3,
    }
    if encode_only:
      if len(layers) > 0:
        feats = []
        for layer, key in enumerate(out):
          if layer in layers:
            feats.append(out[key])
        return feats
      else:
        return out['relu3_1']
    return out

class PerceptualLoss(nn.Module):

  def __init__(self, weights=[0.0, 0.0, 1.0, 0.0, 0.0]):
    super(PerceptualLoss, self).__init__()
    self.add_module('vgg', VGG16())
    self.criterion = nn.L1Loss()
    self.weights = weights

  def __call__(self, x, y):
    # Compute features
    x_vgg, y_vgg = self.vgg(x), self.vgg(y)

    content_loss = 0.0
    content_loss += self.weights[0] * self.criterion(x_vgg['relu1_2'], y_vgg['relu1_2']) if self.weights[0] > 0 else 0
    content_loss += self.weights[1] * self.criterion(x_vgg['relu2_2'], y_vgg['relu2_2']) if self.weights[1] > 0 else 0
    content_loss += self.weights[2] * self.criterion(x_vgg['relu3_3'], y_vgg['relu3_3']) if self.weights[2] > 0 else 0
    content_loss += self.weights[3] * self.criterion(x_vgg['relu4_3'], y_vgg['relu4_3']) if self.weights[3] > 0 else 0
    content_loss += self.weights[4] * self.criterion(x_vgg['relu5_3'], y_vgg['relu5_3']) if self.weights[4] > 0 else 0

    return content_loss


##################################################################################
# Encoder and Decoders
##################################################################################
class ResEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ResEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class AdaEncoder(nn.Module):
    def __init__(self, input_dim, dim, norm, activ, pad_type):
        super(AdaEncoder, self).__init__()
        kernel_size = 3
        stride = 1
        # self.conv1 = ConvLayer(input_dim, dim, kernel_size, stride)
        # self.conv1 = Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)
        self.conv1 = Conv2dBlock(input_dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.conv2 = Conv2dBlock(2 * dim, 4 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.DB1 = DenseBlock(4 * dim, kernel_size, stride)
        self.DB2 = ResBlocks_enc(4 * dim, 253, norm, activation=activ, pad_type=pad_type)

        self.transformer = ViT(image_size=160, patch_size=16, dim=256, depth=12, heads=16, mlp_dim=1024, dropout=0.1,
                               emb_dropout=0.1)
        dim *= 16
        self.output_dim = dim

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x_DB1 = self.DB1(x2)
        # print(x_DB1.shape)
        x_DB2 = self.DB2(x2)
        x_DB = (x_DB1 + x_DB2) / 2
        x_2 = self.transformer(x)
        x_3 = torch.cat((x_DB, x_2), dim=1)
        return x_3

class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='tanh', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activation='relu', pad_type=pad_type)]
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2, mode='nearest'),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation='relu', pad_type=pad_type)]
            self.model.append(SingleModalAttentionLayer(dim // 2))
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class ResBlocks_enc(nn.Module):
    def __init__(self, dim, dim_out, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks_enc, self).__init__()
        self.model = []
        self.model += [ResBlock_enc(dim, dim_out, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model = nn.Sequential(*self.model)
        self.final = LinearBlock(dim, output_dim, norm='none', activation='none') # no output activations

    def forward(self, x):
        self.feature = self.model(x.view(x.size(0), -1))
        out = self.final(self.feature)
        return out#self.model(x.view(x.size(0), -1))

##################################################################################
# Basic Blocks
##################################################################################
class DResBlock(nn.Module):  # extended resblock with different in/ out dimension
    def __init__(self, in_dim, out_dim, kw=4, norm='sn', activation='relu', pad_type='zero', downsample=None, padding=0):
        super(DResBlock, self).__init__()
        self.downsample = downsample
        self.in_dim, self.out_dim = in_dim, out_dim
        self.mid_dim = out_dim
        self.conv1 = Conv2dBlock(self.in_dim, self.mid_dim, kw, 1, norm=norm, activation=activation,
                                 pad_type=pad_type, padding=padding)
        self.conv2 = Conv2dBlock(self.mid_dim, self.out_dim, kw, 1, norm=norm, activation='none', pad_type=pad_type,
                                 padding=padding)
        self.learnable_sc = (in_dim != out_dim)
        if self.learnable_sc:
            self.conv_sc = Conv2dBlock(self.in_dim, self.out_dim, 1, 1, pad_type=pad_type, norm=norm)

    def shortcut(self, x):
        if self.downsample:
            x = self.downsample(x)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return x

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.conv2(h)
        if self.downsample:
            h = self.downsample(h)
        return h + self.shortcut(x)

class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = in_channels - 1
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def*2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out

class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = F.relu(out, inplace=True)
        return out

class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out = out + residual
        return out

class ResBlock_enc(nn.Module):
    def __init__(self, dim, dim_out, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock_enc, self).__init__()
        self.dense_conv_2_1 = Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)
        self.dense_conv_2_2 = Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)
        self.dense_conv_2_3 = Conv2dBlock(dim, dim_out, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)

    def forward(self, x):
        residual = x
        out_1 = self.dense_conv_2_1(residual)
        out_2 = self.dense_conv_2_2(out_1)
        out_3 = self.dense_conv_2_3(residual+out_2)
        out = out_3
        return out

class SingleModalAttentionLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SingleModalAttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x + x * y.expand_as(x)

class MultiModalAttentionLayer(nn.Module):
    def __init__(self, modal_num=3, input_size=40, reduction=16):
        super(MultiModalAttentionLayer, self).__init__()
        self.modal_num = modal_num
        input_channel = input_size**2 * modal_num

        self.fc = nn.Sequential(
            nn.Linear(input_channel, input_channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_channel//reduction, modal_num, bias=False),
            nn.Sigmoid()
        )
        self.sm = nn.Softmax(dim=2)
        self.l1 = nn.Linear(256 * 3, 256)

    def forward(self, feature_list):
        b, c, h, w = feature_list[0].size()
        px = [x.view(b, c, -1) for x in feature_list]
        px = torch.cat(px, dim=-1)
        attention = self.fc(px)
        attention = self.sm(attention)
        attention = torch.transpose(attention, 1, 2)

        x = torch.stack(feature_list, dim=1)
        out = x * attention[:, :, :, None, None]
        out = torch.sum(out, dim=1)

        for i in range(len(feature_list)):
            feature_list[i] = feature_list[i] * 0.9 + out * 0.1

        return torch.cat(feature_list, dim=1)

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=1, dim_head=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim)
        )
        self.dim = dim
        self.patch_size = patch_size
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.downsample = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        x = self.transformer(x)
        x = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.patch_size, h=10, c=1)(x)
        x = self.downsample(x)
        return x

##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)



