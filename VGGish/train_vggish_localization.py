import argparse
import glob
import os
from pathlib import Path
import random
import sys

import IPython
import numpy as np
from scipy.spatial.transform import Rotation
import soundfile as sf
import torch
import torchaudio.functional as F

import config
sys.path.insert(0, str(config.LIBS_DIR.joinpath("RotationContinuity/sanity_test/code")))
import tools
import models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute VGGish features for input path')
    parser.add_argument('audio_dir', help='File pattern for audio file')
    parser.add_argument('centroid_dir', help='File pattern for centroid file')
    parser.add_argument('--error_path', help='File pattern for saving errors')
    parser.add_argument('--save_path', help='Path for saving model state')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='Size of batches')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of batch iterations')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained VGGish weights')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false', help='Do not use pretrained VGGish weights')
    parser.set_defaults(pretrained=False)
    parser.add_argument('--wavelet', action='store_true', help='Use Wavelet architecture')
    parser.set_defaults(wavelet=False)
    parser.add_argument('--resnet1d', action='store_true', help='Use ResNet1D architecture')
    parser.set_defaults(resnet1d=False)
    parser.add_argument('--complex_vggish', action='store_true', help='Use VGGish architecture with complex spectrogram input')
    parser.set_defaults(complex_vggish=False)
    parser.add_argument('--multi_chan', action='store_true', help='Use multichannel VGGish') #new
    parser.set_defaults(multi_chan=False)
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--normalized', action='store_true', help='Use normalized spectrogram with normalized/clipped magnitude and sin/cos channels')
    parser.add_argument('--raw', dest='normalized', action='store_false', help='Use raw complex spectrogram in two channels')
    parser.set_defaults(normalized=False)
    parser.add_argument('--mic_id', type=int, default=-1, help='ID of microphone to select')
    parser.add_argument('--num_channels', type=int, default=1, help='Number of microphones to use per example')
    parser.add_argument('--music', action='store_true', help='If we are using music')
    parser.add_argument('--resample', action='store_true', help='If we are resampling')
    parser.add_argument('--silence', action='store_true', help='If we testing it out on silence')

    #More arguments
    parser.add_argument('--darkroom', action='store_true', help='dr', default=False)
    parser.add_argument('--living', action='store_true', help='living')
    parser.add_argument('--conference', action='store_true', help='conference')

    #Generalize Path
    parser.add_argument('--g_audio_dir', help='File pattern for audio generalize', default=None)
    parser.add_argument('--g_centroid_dir', help='File pattern for centroid file generalize', default=None)


    parser.add_argument('--num_train', type=int, help='number of training examples', default=None)


    args = parser.parse_args()    
    rand = random.Random(args.seed)
    
    np.random.seed(0)
    deconv = np.load(args.audio_dir)
    print("Deconv Loaded")
    centroid = np.load(args.centroid_dir)
    print("Centroid Loaded")

    if args.mic_id >= 0:
        mic_ids = np.load(os.path.join(args.data_dir, "micID.npy"))
        mask = (mic_ids == args.mic_id)
        deconv = deconv[mask, ...]
        centroid = centroid[mask, ...]
    
    if args.num_channels < 10:
        if args.darkroom:
            if args.num_channels == 4:
                mic_indices = [0, 5, 6, 9]
            if args.num_channels == 2:
                mic_indices = [0, 6]
            if args.num_channels == 1:
                mic_indices = [0]

        elif args.conference:
            if args.num_channels == 4:
                mic_indices = [0, 1, 5, 6]
            if args.num_channels == 2:
                mic_indices = [1, 6]
            if args.num_channels == 1:
                mic_indices = [1]

        elif args.living:
            if args.num_channels == 4:
                mic_indices = [8, 0, 5, 3]
            if args.num_channels == 2:
                mic_indices = [0, 5]
            if args.num_channels == 1:
                mic_indices = [0]
        else:
            exit()
    else:
        mic_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    deconv = deconv[:, mic_indices, :]

    print(deconv.shape)
    if args.music or args.silence:
        print("dividing to float")
        deconv = deconv/32768.0

    print(centroid.shape)

    train_indices = np.load("indices/train_indices.npy")

    if args.num_train is not None:
        train_indices = np.random.choice(train_indices, args.num_train,replace=False)

    valid_indices = np.load("indices/valid_indices.npy")
    test_indices = np.load("indices/test_indices.npy")

    #Centroids
    train_xy = centroid[train_indices]
    valid_xy = centroid[valid_indices]
    test_xy = centroid[test_indices]

    train_mean = np.mean(train_xy, axis=0)
    train_std = np.std(train_xy, axis=0)

    print(train_std)
    print(np.max(train_xy, axis=0))
    print(np.min(train_xy, axis=0))
    train_xy = (train_xy - train_mean) / (train_std + 1e-8)
    valid_xy = (valid_xy - train_mean) / (train_std + 1e-8)
    test_xy =  (test_xy - train_mean) / (train_std + 1e-8)


    #Code insertion ends
    epsilon = 1e-2
    norm_val_min = np.min(np.concatenate((train_xy, valid_xy), axis=0))
    norm_val_range = np.max(np.concatenate((train_xy, valid_xy), axis=0)) - norm_val_min

    def postprocess_net_output(output):
        output[:, :2] = norm_val_range * ((torch.tanh(output[:, :2]) * (1 + epsilon)) + 1) / 2 + norm_val_min
        return output
    

    train_std_cuda = torch.Tensor(train_std).cuda()
    train_mean_cuda = torch.Tensor(train_mean).cuda()

    def unnormalize(xy):
        return xy*(train_std_cuda + 1e-8) + train_mean_cuda

    def resample(audio, ir=48000, tr=16000):

        resampled_waveform = F.resample(
        audio,
        ir,
        tr,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="kaiser_window",
        beta=14.769656459379492,
        )
        return resampled_waveform


    train_waves = deconv[train_indices, :]
    # 30950 seems to be the rough cutoff after which vggish treats the input as two examples.
    valid_waves = deconv[valid_indices, :]
    #Test Waves
    test_waves = deconv[test_indices, :]

    offset = 0
    if args.music:
        offset = 5*48000
    if args.silence:
        offset = 12*48000

    precutoff = 92850
    train_waves = train_waves[..., (offset):(offset+precutoff)]
    valid_waves = valid_waves[..., (offset):(offset+precutoff)]
    test_waves = test_waves[..., (offset):(offset+precutoff)]

    train_waves = torch.Tensor(train_waves).cuda()
    train_xy = torch.Tensor(train_xy).cuda()

    valid_waves = torch.Tensor(valid_waves).cuda()
    valid_xy = torch.Tensor(valid_xy).cuda()

    test_waves = torch.Tensor(test_waves).cuda()
    test_xy = torch.Tensor(test_xy).cuda()


    if args.resample:
        print("Resampling")
        train_waves = resample(train_waves)
        valid_waves = resample(valid_waves)
        test_waves = resample(test_waves)

    vggish_cutoff = config.N_STEPS if (args.resnet1d or args.complex_vggish or args.wavelet) else 30950
    
    if args.multi_chan:
        vggish_cutoff = 15475

    train_waves = train_waves[..., :vggish_cutoff]
    valid_waves = valid_waves[..., :vggish_cutoff]
    test_waves = test_waves[..., :vggish_cutoff]

    out_channels = 2
    
    print("Done Resampling")

    if args.resnet1d:
        net = models.get_resnet1d_model(out_channels=out_channels)
        print('Using ResNet-based model')
    elif args.complex_vggish:
        clip_value = 14 if args.normalized else None
        in_channels = None
        if args.num_channels > 1:
            in_channels = int((3 if args.normalized else 1) * args.num_channels) #modified
        net = models.CustomVGGish(in_channels=in_channels, out_channels=out_channels, normalized=args.normalized, clip_value=clip_value)
        print('Using VGGish-based model with complex inputs')


    elif args.wavelet:
        clip_value = 14 if args.normalized else None
        in_channels = None
        if args.num_channels > 1:
            in_channels = int((3 if args.normalized else 2) * args.num_channels)
        net = models.CustomVGGish(in_channels=in_channels, out_channels=out_channels, normalized=args.normalized, clip_value=clip_value)
        print('Using WaveletVGGish-based model with complex inputs')
    elif args.multi_chan:
        net = models.CustomVGGish2(in_channels=args.num_channels, out_channels=out_channels)

    else:

        train_waves = torch.mean(train_waves, dim=1)
        valid_waves = torch.mean(valid_waves, dim=1)
        test_waves = torch.mean(test_waves, dim=1)
        
        net = models.get_finetune_model(pretrained=args.pretrained, frozen=args.pretrained, out_channels=out_channels)
        print('Using VGGish-based model')
    
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print('Total parameters: %i'%total_params)
    print('Trainable parameters: %i'%trainable_params)
    
    xy_loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999995)
    
    print("Number of samples per epoch:", train_waves.shape[0])
    
    N_iter = int(train_waves.shape[0] / args.batch_size)
    train_losses = []
    train_xy_losses = []
    valid_losses = []
    valid_xy_losses = []
    step_count = 0
    for n in range(args.num_epochs):
        print('Reshuffling for Epoch %i'%n, flush=True)
        rand_idx = np.random.permutation(train_waves.shape[0])
        net.train()
        optimizer.zero_grad()
        for i in range(N_iter):
            curr_idx = rand_idx[i*args.batch_size:(i+1)*args.batch_size]
            net_out = net(train_waves[curr_idx, :])
            #print("net out shape")
            #print(net_out.shape)
            results = postprocess_net_output(net_out)
            #print("results shape")
            #print(results.shape)
            xy_loss = xy_loss_fn(results[:, :2], train_xy[curr_idx, :2])
            loss = xy_loss
            optimizer.zero_grad()
            loss.backward()
            train_loss = loss.item()
            #print(train_loss)
            train_losses.append((step_count, train_loss))
            train_xy_losses.append((step_count, xy_loss.item()))
            step_count+=1
            optimizer.step()
            scheduler.step()
        # print(results)
        # print(train_meta[curr_idx, :2])

        net.eval()
        valid_loss_xy_arr = np.zeros(valid_waves.shape[0], dtype=np.float32)
        valid_loss_arr = np.zeros(valid_waves.shape[0], dtype=np.float32)
        for i in range(valid_waves.shape[0]):
            with torch.no_grad():
                results = torch.squeeze(postprocess_net_output(net(torch.unsqueeze(valid_waves[i, :], axis=0)).view(-1, 1)))
            xy_loss = xy_loss_fn(results[:2], valid_xy[i, :2])
            valid_loss_xy_arr[i] = xy_loss.item()
            loss = xy_loss
            valid_loss_arr[i] = loss.item()
        valid_xy_loss = np.mean(valid_loss_xy_arr)
        valid_loss = np.mean(valid_loss_arr)
        print('Validation XY Loss: %0.3f'%valid_xy_loss)
        print('Validation Loss: %0.3f'%valid_loss)
        valid_losses.append((step_count, valid_loss))
        valid_xy_losses.append((step_count, valid_xy_loss))



        if not os.path.exists(args.error_path):
            os.makedirs(args.error_path)

        np.save(os.path.join(args.error_path, 'train_losses.npy'), np.array(train_losses, dtype=np.float32))
        np.save(os.path.join(args.error_path, 'valid_losses.npy'), np.array(valid_losses, dtype=np.float32))

        #Iterate through test
        test_errors = np.zeros(test_waves.shape[0], dtype=np.float32)

        for i in range(test_waves.shape[0]):
            with torch.no_grad():
                results = torch.squeeze(postprocess_net_output(net(torch.unsqueeze(test_waves[i, :], axis=0)).view(-1, 1)))
            
            
            test_errors[i] = torch.norm(unnormalize(results[:2]) - unnormalize(test_xy[i, :2])).item()

        print("TEST ERROR")
        print(test_errors)
        
        print("MEAN TEST ERROR",flush=True)
        print(np.mean(test_errors))
        print("MED TEST ERROR")
        print(np.median(test_errors))
        print("STD TEST ERROR")
        print(np.std(test_errors))

        np.save(os.path.join(args.error_path, 'test_errors.npy'), np.array(test_errors, dtype=np.float32))        


        torch.save({
            'epoch': n,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'train_xy_losses': train_xy_losses,
            'valid_losses': valid_losses,
            'valid_xy_losses': valid_xy_losses,
            'train_mean': train_mean,
            'train_std': train_std,
            'norm_val_min':norm_val_min,
            'norm_val_range':norm_val_range,
            'lr': args.lr,
            }, args.save_path)
    
    #Generalize Path
    if args.g_audio_dir is not None:
        print("TESTING GENERALIZABILITY")
        deconv = np.load(args.g_audio_dir)
        centroid = np.load(args.g_centroid_dir)
        print("loaded")
        deconv = deconv[:800]
        centroid = centroid[:800]
        centroid = torch.Tensor(centroid).cuda()

        offset = 0
        
        if args.music:
            offset = 5*48000

        if args.pretrained:
            deconv = deconv[..., (offset):(offset+123800)]
            deconv = torch.Tensor(deconv).cuda()
            deconv = resample(deconv)
            deconv = deconv[..., mic_indices, :30950]
            deconv = torch.mean(deconv, dim=1)
        elif args.multi_chan:
            deconv = deconv[:, mic_indices, (offset):(offset+15475)]
            deconv = torch.Tensor(deconv).cuda()

        if args.music:
            deconv = deconv/32768.0

        net.eval()
        test_errors = np.zeros(deconv.shape[0], dtype=np.float32)

        for i in range(deconv.shape[0]):
            with torch.no_grad():
                results = torch.squeeze(postprocess_net_output(net(torch.unsqueeze(deconv[i, :], axis=0)).view(-1, 1)))

            test_errors[i] = torch.norm(unnormalize(results[:2]) - centroid[i, :2]).item()
        print("GENERALIZE - Mean, med, std")
        print(test_errors)
        print(np.mean(test_errors))
        print(np.median(test_errors))
        print(np.std(test_errors))