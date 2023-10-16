import argparse
import os
import random
import sys
import numpy as np
import torch
import torchaudio.functional as F
import config
import models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute VGGish features for input path')
    parser.add_argument('audio_dir', help='File pattern for audio file')
    parser.add_argument('labels_dir', help='File pattern for labels')
    parser.add_argument('--error_path', help='File pattern for saving errors',default="errors")
    parser.add_argument('--save_path', default='mode.pt', help='Path for saving model state')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='Size of batches')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of batch iterations')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained VGGish weights')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false', help='Do not use pretrained VGGish weights')
    parser.set_defaults(pretrained=False)
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

    #Extra arguments
    parser.add_argument('--treated', action='store_true', help='dr', default=False)
    parser.add_argument('--livingroom', action='store_true', help='livingroom')
    parser.add_argument('--conference', action='store_true', help='conference')

    #Generalize Path
    parser.add_argument('--g_audio_dir', help='File pattern for audio generalize', default=None)
    parser.add_argument('--g_labels_dir', help='File pattern for labels generalize', default=None)


    #number of categories:
    parser.add_argument('--num_categories', type=int, help='number of classification categories')
    parser.add_argument('--train_indices', type=str,help='File pattern for train')
    parser.add_argument('--valid_indices', type=str, help='File pattern for valid')
    parser.add_argument('--test_indices', type=str, help='File pattern for test')

    #Empty, if binary
    parser.add_argument('--empty_dir', type=str, help='Empty Audio Path, if Binary', default=None)
    args = parser.parse_args()    
    rand = random.Random(args.seed)
    
    deconv = np.load(args.audio_dir)

    if args.empty_dir is not None:
        deconv2 = np.load(args.empty_dir)
        deconv = np.concatenate((deconv, deconv2),axis=0)

    print("Deconv Loaded")
    labels = np.load(args.labels_dir)
    print("Labels Loaded")
    labels = torch.Tensor(labels).cuda()

    if args.num_channels < 10:
        if args.treated:
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

        elif args.livingroom:
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



    if args.num_categories > 1:
        labels = torch.nn.functional.one_hot(labels.long(), args.num_categories)
        labels = labels.float()
    else:
        labels = torch.unsqueeze(labels, dim=-1)
    print("onehot generated")
    print(labels.shape)

    #Loading train indices
    train_indices = np.load(args.train_indices)
    valid_indices = np.load(args.valid_indices)
    test_indices = np.load(args.test_indices)

    train_labels = labels[train_indices]
    valid_labels = labels[valid_indices]
    test_labels = labels[test_indices]

    print(deconv.shape)
    if args.music or args.silence:
        print("dividing to float")
        deconv = deconv/32768.0

    print(labels.shape)

    if args.num_categories > 1:
        sm = torch.nn.Softmax(dim=-1)
    else:
        sm = torch.nn.Sigmoid()

    def postprocess_net_output(output):
        
        if args.num_categories > 1:
            output[..., :args.num_categories] = sm(output[..., :args.num_categories])
        else:
            output[..., :args.num_categories] = sm(50*output[..., :args.num_categories])
            
        return output
    

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
    train_labels = torch.Tensor(train_labels).cuda()

    valid_waves = torch.Tensor(valid_waves).cuda()
    valid_labels = torch.Tensor(valid_labels).cuda()

    test_waves = torch.Tensor(test_waves).cuda()
    test_labels = torch.Tensor(test_labels).cuda()


    if args.resample:
        print("Resampling")
        train_waves = resample(train_waves)
        valid_waves = resample(valid_waves)
        test_waves = resample(test_waves)

    vggish_cutoff = config.N_STEPS if (args.resnet1d or args.complex_vggish) else 30950
    
    if args.multi_chan:
        vggish_cutoff = 15475

    train_waves = train_waves[..., :vggish_cutoff]
    valid_waves = valid_waves[..., :vggish_cutoff]
    test_waves = test_waves[..., :vggish_cutoff]

    out_channels = args.num_categories
    
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
    
    if args.num_categories > 1:
        loss_fcn = torch.nn.CrossEntropyLoss()
    else:
        loss_fcn = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999995)
    
    print("Number of samples per epoch:", train_waves.shape[0])
    
    N_iter = int(train_waves.shape[0] / args.batch_size)
    train_losses = []
    step_count = 0
    for n in range(args.num_epochs):
        print('Reshuffling for Epoch %i'%n, flush=True)
        rand_idx = np.random.permutation(train_waves.shape[0])
        net.train()
        optimizer.zero_grad()
        for i in range(N_iter):
            curr_idx = rand_idx[i*args.batch_size:(i+1)*args.batch_size]
            net_out = net(train_waves[curr_idx, :])
            results = postprocess_net_output(net_out)
            loss = loss_fcn(results[:, :args.num_categories], train_labels[curr_idx,:args.num_categories])
            optimizer.zero_grad()
            loss.backward()
            train_loss = loss.item()
            train_losses.append((step_count, train_loss))
            step_count+=1
            optimizer.step()
            scheduler.step()

        net.eval()

        if not os.path.exists(args.error_path):
            os.makedirs(args.error_path)

        np.save(os.path.join(args.error_path, 'train_losses.npy'), np.array(train_losses, dtype=np.float32))

        #Iterate through test
        test_acc = np.zeros(test_waves.shape[0], dtype=np.float32)

        for i in range(test_waves.shape[0]):
            with torch.no_grad():
                results = torch.squeeze(postprocess_net_output(net(torch.unsqueeze(test_waves[i, :], axis=0))))
            

            if args.num_categories > 1:
                prediction = torch.argmax(results)
                ground_truth = torch.argmax(test_labels[i])
            else:
                prediction = torch.round(results).item()
                ground_truth = test_labels[i]

            if prediction == ground_truth: 
                test_acc[i] = 1
            else:
                test_acc[i] = 0
            

        print("TEST ACCURACIES")
        print(test_acc)
        
        print("MEAN TEST ACCURACY", flush=True)
        print(np.mean(test_acc), flush=True)
        np.save(os.path.join(args.error_path, 'test_acc.npy'), np.array(test_acc, dtype=np.float32))        

        torch.save({
            'epoch': n,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr': args.lr,
            }, args.save_path)
    
    #Generalize Path
    if args.g_audio_dir is not None:
        print("TESTING GENERALIZABILITY")
        deconv = np.load(args.g_audio_dir)
        labels = np.load(args.g_labels_dir)
        print("loaded")
        deconv = deconv[:800]
        labels = labels[:800]
        labels = torch.Tensor(labels).cuda()

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
        test_acc = np.zeros(deconv.shape[0], dtype=np.float32)

        for i in range(deconv.shape[0]):
            with torch.no_grad():
                results = torch.squeeze(postprocess_net_output(net(torch.unsqueeze(test_waves[i, :], axis=0)).view(-1, 1)))
            
            prediction = torch.argmax(results)
            ground_truth = torch.argmax(test_labels[i])
            if prediction == ground_truth: 
                test_acc[i] = 1
            else:
                test_acc[i] = 0
            

        print("GENERALIZE - Mean")
        print(test_acc)
        print(np.mean(test_acc))

