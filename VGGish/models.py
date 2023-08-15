import sys

import numpy as np
import torch
from torchvggish.vggish import VGG, VGGish, make_layers
from torchvggish.vggish_input import waveform_to_examples
import torchvggish.mel_features
import torchvggish.vggish_params



import config
sys.path.insert(0, str(config.LIBS_DIR.joinpath('resnet1d')))
from resnet1d import ResNet1D


#Wavelet stuff
from pytorch_wavelets import DWTForward
#from wavelets_pytorch.transform import WaveletTransformTorch 
from scipy import signal
import pywt
import morlwavelet


# TODO(sclarke): This was just copied from the hubconf file in the original repo, this should use an external reference.
vggish_model_urls = {
    'vggish': 'https://github.com/harritaylor/torchvggish/'
              'releases/download/v0.1/vggish-10086976.pth',
    'pca': 'https://github.com/harritaylor/torchvggish/'
           'releases/download/v0.1/vggish_pca_params-970ea276.pth'
}

def vggish(pretrained=True, **kwargs):
    model = VGGish(urls=vggish_model_urls, pretrained_net=pretrained, postprocess=pretrained, **kwargs)
    return model

class VGGModel(torch.nn.Module):
        
        def __init__(self, fs=16000, pretrained=True):
            super(VGGModel, self).__init__()
            # self.vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
            self.vggish = vggish(pretrained=pretrained)
            self.vggish.eval()
            self.fs = fs
            
        def forward(self, x):
            x = (self.vggish.forward(torch.unsqueeze(x, axis=1), fs=self.fs) / 127) - 1
            # x = (self.vggish.forward(np.expand_dims(x, axis=1), fs=self.fs) / 127) - 1
            return x    
    
def get_finetune_model(pretrained=True, frozen=True, out_channels=2, dropout=False, in_channels=1):
    # TODO(sclarke): This pretrained model is based on 16kHz samples, and automatically downsamples input audio to 16k
    vggish_model = VGGModel(fs=config.PSEUDO_SAMPLE_RATE, pretrained=pretrained)
    if pretrained and frozen:
        for param in vggish_model.parameters():
            param.requires_grad = False
    p_dropout = 0.5 if dropout else 0.0
    finetune_model = torch.nn.Sequential(
        vggish_model,
        torch.nn.Linear(in_features=128, out_features=256, bias=True),
        torch.nn.ReLU(),
        torch.nn.Dropout(p_dropout),
        torch.nn.Linear(in_features=256, out_features=256, bias=True),
        torch.nn.ReLU(),
        torch.nn.Dropout(p_dropout),
        # torch.nn.Linear(in_features=256, out_features=256, bias=True),
        # torch.nn.ReLU(),
        torch.nn.Linear(in_features=256, out_features=out_channels, bias=True)
    ).cuda()
    
    return finetune_model

class ResNet1DModel(torch.nn.Module):
        
        def __init__(self, out_channels=2):
            super(ResNet1DModel, self).__init__()
            self.resnet = ResNet1D(in_channels=1,
                                    base_filters=64,
                                    kernel_size=16,
                                    stride=2,
                                    groups=32,
                                    n_block=36,
                                    n_classes=out_channels,
                                    downsample_gap=6,
                                    increasefilter_gap=12,
                                    use_bn=True,
                                    use_do=True,
                                    verbose=False).cuda()
            
        def forward(self, x):
            x = self.resnet.forward(torch.unsqueeze(x, axis=1))
            return x  

def get_resnet1d_model(out_channels=2):
    resnet_model = ResNet1DModel(out_channels=out_channels)
    return resnet_model

class CustomVGGish(VGG):
    def __init__(self, n_fft=256, hop_length=64, window_length=192, in_channels=None, out_channels=2, p_dropout=0.0, normalized=False, clip_value=None, device=None,):
        if not in_channels:
            #modified
            in_channels = 3 if normalized else 2 #2
            print('YO')
        super().__init__(make_layers(in_channels=in_channels, layer_desc=[64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, "M"]), cnn_height=4, cnn_width=4)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.to(self.device)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_length = window_length
        self.normalized = normalized
        self.clip_value = clip_value
        self.mlp = torch.nn.Sequential(
                            torch.nn.Linear(in_features=128, out_features=256, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(p_dropout),
                            torch.nn.Linear(in_features=256, out_features=256, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(p_dropout),
                            torch.nn.Linear(in_features=256, out_features=out_channels, bias=True)
                            ).cuda()

    def forward(self, x):
        x = self._preprocess(x)
        x = VGG.forward(self, x)
        x = self.mlp(x)
        return x

    def _preprocess(self, x):
    #The normalize flag will essentially determine if the preprocessing is like that in pretrained, or not.

        if isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
            reshape = False
            original_size = x.size()
            if x.dim() > 2:
                x = x.view(-1, x.size()[-1])
                reshape = True
            
            
            x = torch.stft( x,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.window_length,
                            window=torch.hann_window(self.window_length).to(x.device),
                            return_complex=self.normalized
                            )

            #print("XSIZE")
            #print(x.size())

            if self.normalized:
                mag = torch.abs(x)
                if self.clip_value:
                    mag = torch.clamp(mag, max=self.clip_value) / self.clip_value
                angle = torch.angle(x)
                x = torch.stack((mag, torch.sin(angle), torch.cos(angle)), dim=1)
                #print("Normalized size before reshape")
                #print(x.size())
            else:
                x = torch.norm(x, dim=-1)
                x = torch.unsqueeze(x, dim=1)
                    

            if reshape:
                # print("orig siez")
                # print(original_size)
                # print("XSIZE")
                # print(x.size())
                # print(((original_size[0], original_size[1]*x.size()[1], x.size()[2], x.size()[3])))
                x = x.view(original_size[0], original_size[1]*x.size()[1], x.size()[2], x.size()[3])
                # print("XSIZE after view")
                # print(x.size())
                
        else:
            raise AttributeError
        return x

    def _postprocess(self, x):
        self.pproc.train(self.training)
        return self.pproc(x)



class CustomVGGish2(VGG):
    def __init__(self, in_channels=10, out_channels=2, device=None, p_dropout=0.0):
        super().__init__(make_layers(in_channels=in_channels))


        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.device = device
        self.to(self.device)


        self.mlp = torch.nn.Sequential(
                            torch.nn.Linear(in_features=128, out_features=256, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(p_dropout),
                            torch.nn.Linear(in_features=256, out_features=256, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(p_dropout),
                            torch.nn.Linear(in_features=256, out_features=out_channels, bias=True)
                            ).cuda()

    def forward(self, x):
        x = self._preprocess(x)
        x = VGG.forward(self, x)
        x = self.mlp(x)
        return x

    def _preprocess(self, x):
    #The normalize flag will essentially determine if the preprocessing is like that in pretrained, or not.
        
        original_size = x.size()

        if x.dim() > 2:
            x = x.view(-1, x.size()[-1])
            reshape = True

        # Compute log mel spectrogram features.
        x = torchvggish.mel_features.log_mel_spectrogram(
            x,
            audio_sample_rate=torchvggish.vggish_params.SAMPLE_RATE,
            log_offset=torchvggish.vggish_params.LOG_OFFSET,
            window_length_secs=torchvggish.vggish_params.STFT_WINDOW_LENGTH_SECONDS,
            hop_length_secs=torchvggish.vggish_params.STFT_HOP_LENGTH_SECONDS,
            num_mel_bins=torchvggish.vggish_params.NUM_MEL_BINS,
            lower_edge_hertz=torchvggish.vggish_params.MEL_MIN_HZ,
            upper_edge_hertz=torchvggish.vggish_params.MEL_MAX_HZ)
            
        if reshape:
            #print("orig siez")
            #print(original_size)
            #print("XSIZE")
            #print(x.size())
            #print(((original_size[0], original_size[1], x.size()[1], x.size()[2])))
            x = x.view(original_size[0], original_size[1], x.size()[1], x.size()[2])
            #print("XSIZE after view")
            #print(x.size())

        return x

            




# class CustomVGGishWavelet(CustomVGGish):
#     def __init__(self, n_fft=256, hop_length=64, window_length=192, in_channels=None, out_channels=2, p_dropout=0.0, normalized=False, clip_value=None, device=None):
#         super().__init__(n_fft, hop_length, window_length, in_channels, out_channels, p_dropout, normalized, clip_value, device)
#     def _preprocess(self, x):
#         if isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
#             reshape = False
#             original_size = x.size()
#             if x.dim() > 2:
#                 x = x.view(-1, x.size()[-1])
#                 reshape = True
            

#             #xfm = DWTForward(J=3, mode='zero', wave='db3')
#             #print(x.size())
#             #x= xfm(x)
#             #x = torch.stack(xL, xH)

#             #x = x.cpu().detach().numpy()
#             #widths = np.arange(1, 130)
#             #cwtmatr, freqs = pywt.cwt(x, widths, 'morl')
#             #x = cwtmatr
#             #print(x.shape)
#             #x = torch.from_numpy(x)
#             #x = x.cuda()


#             pycwt = morlwavelet.CWT(dt=1/48000, dj=0.036, hop_length=64)
#             x = torch.unsqueeze(x, 1)
#             print(x.shape)
#             x = pycwt(x)
#             x = torch.squeeze(x)
#             del pycwt
#             #wavelets = pywt.wavedec(x, 'db8', level=128)
#             #torch_list = [torch.from_numpy(w)[:,:15] for w in wavelets]
#             #x = torch.stack(torch_list, axis=1)
    
#             print("X SIZE " + str(x.size()))

#             if self.normalized:
#                 mag = torch.abs(x)
#                 if self.clip_value:
#                     mag = torch.clamp(mag, max=self.clip_value) / self.clip_value
#                 angle = torch.angle(x)
#                 x = torch.stack((mag, torch.sin(angle), torch.cos(angle)), dim=1)
#             else:
#                 x = torch.transpose(x, 1, 3)
#             if reshape:
#                 #x = x.view(original_size[0], original_size[1]*x.size()[1], x.size()[2], x.size()[3])
#                 print("orig siez")
#                 print(original_size)
#                 x = x.view(original_size[0], original_size[1]*x.size()[1], x.size()[2], x.size()[3])

#         else:
#             raise AttributeError
#         print("RET " + str(x.shape))
#         return x