#!/usr/bin/env python
import pickle
import torch
import common.state
import common.datasets
from common.log import log
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import lpips
import time
import os, psutil
from tqdm import tqdm
import random
random.seed(0)
#random.seed(1)
import argparse
import resnet
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

plt.tight_layout()

import linecache
import os
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))



parser = argparse.ArgumentParser()
parser.add_argument('model',  type=str, help='Model name')
parser.add_argument('dataset',  type=str, help='dataset name')
parser.add_argument('--dist',  type=str, default="lpips_alex", help='distance function')
# parser.add_argument('--dist',  type=str, default="lpips_alex" help='distance function')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--step_size', type=float, default=5e-2,
                    help='batch size')
parser.add_argument('--iters', type=int, default=100,
                    help='batch size')
parser.add_argument('--num_images', type=int, default=100,
                    help='number of images per class')
parser.add_argument('--examples', action='store_true',
                    help='example images')
parser.add_argument('--get_widths', action='store_true',
                    help='Get widths')
parser.add_argument('--full_eval', action='store_true',
                    help='example images')
parser.add_argument('--show_grad', action='store_true',
                    help='Grad Cam')
parser.add_argument('--use_logit_init', action='store_true',
                    help='Grad Cam')
parser.add_argument('--desc', type=str, default="",
                    help='number of images per class')



args = parser.parse_args()
if args.get_widths:
    args.batch_size = 1
    args.full_eval = False


def prod(x):
    p = 1
    for i in x:
        p = p*i
    return p

def progress(batch, batches, epoch=None):
    if batch == 0:
        if epoch is not None:
            log(' %d .' % epoch, end='')
        else:
            log(' .', end='')
    else:
        log('.', end='', context=False)

    if batch == batches - 1:
        log(' done', end="\n", context=False)

def visualize(inp, reshape=False):
    if reshape:
        inp = torch.stack((inp[0], inp[1], inp[2] ), dim=-1)
    return Image.fromarray((255*inp.numpy()).astype(np.uint8))

def measure_width(img, model, index, dir_vec):
    init_confidence = softmax(model(img.clamp(0,1)))[:, index]
    num_vecs = 256
    rand_vecs = torch.randn(num_vecs, *img.shape[1:]).to(init_confidence.device)
    if torch.linalg.vector_norm(dir_vec) == 0:
        return np.zeros((7, 2))
    dir_vec = dir_vec/torch.linalg.vector_norm(dir_vec)
    a = torch.sum(rand_vecs*dir_vec, dim=(1,2,3))
    rand_vecs = rand_vecs - torch.einsum("ij, jklm -> iklm" ,a.unsqueeze(-1), dir_vec)
#     rand_vecs = rand_vecs.reshape(num_vecs, -1)
#     img = img.reshape(-1)
    rand_vecs = rand_vecs/torch.linalg.vector_norm(rand_vecs, dim=(1,2,3), keepdim=True)
#     distances = [5, 7.5, 10, 12.5, 15, 20]
    distances = [5, 10, 15, 20, 25, 30]
    confs = [(float(init_confidence), float(init_confidence))]
    for i in distances:
        pert_imgs = img + i*rand_vecs
        pert_confs = softmax(model(pert_imgs.clamp(0,1)))[:, index]
#         min_confs, max_confs = float(torch.min(pert_confs)), float(torch.max(pert_confs))
        confs.append(torch.quantile(pert_confs, q=torch.Tensor([0.05, 0.95]).to(pert_confs.device) ).cpu().numpy().tolist())
    confs = np.array(confs)
    return confs

def total_variation_loss(img, weight):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
    tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
    return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

def logit_init(model, img, source_classes, target_img=None):
    if target_img is None:
        target_img = img
    logits_init = model(img)
    logits_shift = torch.tensor([1.0]).cuda()
    logits_shift = logits_shift.requires_grad_(True)
    logits_targ = logits_init+0.1*logits_shift
    epsilon = 40./255
    steps = 200
    pert = (torch.rand_like(target_img).cuda() - 0.5)*2 
    pert = torch.clamp(target_img.data+pert,0.0,1.0)-target_img.data
    pert = pert.clamp(0.0,0.0)
    p_init = pert.clone()
    opt = torch.optim.Adam([pert], lr=10)
    pert.requires_grad = True
    indices = torch.arange(len(target_img))
    pred = model((img).clamp(0,1))
    init_probs = softmax(pred).detach()
    relu = torch.nn.ReLU()
    for step in range(steps):
        logits = model(target_img+pert)
        probs = softmax(logits)
        #loss = torch.norm(logits-logits_targ.data)**2.0 + lpips_fn(img+pert,img_targ).sum()
        #loss = torch.norm(logits-logits_init-logits_shift.data)**2.0
        #loss = nn.CrossEntropyLoss()(logits, torch.LongTensor([0,0]).cuda())
        loss = (torch.norm(relu(init_probs[indices,source_classes]-probs[indices,source_classes]))) #+ 0*(relu(20 - torch.norm(pert)))**2 + 0*(relu(torch.norm(pert) - 40))**2 + 100*total_variation_loss(pert, 1) #+ torch.norm(logits-logits_targ.data)
        print(loss)
        opt.zero_grad()
        loss.backward()
        #flag = torch.max(probs,dim=-1)[0] > 0.9
        #print(step,flag,torch.norm(pert-p_init).item(),loss.data.item())
        opt.step()
        pert.data = torch.clamp(target_img.data+pert.data,0.0 + epsilon,1.0 - epsilon)-target_img.data
        #logits_targ.data = logits_init+logits_shift
#         pert.data.clamp_(-epsilon,epsilon)
        if step in [steps//2,(3*steps)//4]:
            for param_group in opt.param_groups:
                param_group['lr'] /= 10.0
            print(param_group['lr'])
    print("Pert norm:", torch.norm(pert))
    print("Confidence:", probs[indices,source_classes] )
    return target_img.detach(), pert.detach()

def batch_nullspace_attack(dataset, source_classes, model, stepsize, target_image, iterations, device, order=1, inp=None, get_widths=False):
    # target_image = torch.ones(inp.shape, device=device).reshape((-1))
    source_classes = source_classes.to(device)
    if inp is None:
        source_image = common.torch.as_variable(dataset, True)
        source_image = source_image.permute(0, 3, 1, 2)
        #print(source_image.shape)
        inp = source_image.clone().to(device).requires_grad_(True)
    else:
        source_image = inp
        inp = source_image.detach().clone().to(device).requires_grad_(True)
    mask = torch.ones(*inp.shape, dtype=torch.bool).to(device)
    target_image = common.torch.as_variable(target_image, True)
    target_image = target_image.permute(2, 0, 1)
    indices = torch.arange(len(inp))
    pred = model((inp).clamp(0,1))
    init_probs = softmax(pred)
    #print(init_probs)
    init_labels = torch.argmax(init_probs, dim=-1)
    irrel_images = init_labels != source_classes
#     step_size_vec = step_size*torch.ones(inp.shape)
    all_confs = []
    for i in range(iterations):
        pred = model((inp).clamp(0,1))
        probs = softmax(pred)
#        scaling_factors = 1 + 100*torch.sum((probs - init_probs)**2, dim=-1)
#        print(scaling_factors.mean())
        sum_probs = torch.sum(probs[indices,source_classes])
        sum_probs.backward()
        J = inp.grad.reshape(len(probs), prod(inp.shape[1:]))
        v = target_image.flatten() - inp.flatten(start_dim=1)
        null_vec = get_nullspace_projection(J, v).reshape(inp.shape)
#        new_inp = inp + (stepsize/scaling_factors).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*null_vec
        if get_widths and i%6 == 0 and i < 60:
            print(i)
            with torch.no_grad():
                all_confs.append((inp.detach().cpu(), measure_width(inp, model, source_classes, null_vec) ))
        new_inp = inp + stepsize*null_vec
        pred_probs = softmax(model(new_inp.clamp(0,1)))
        pred_labels = torch.argmax(pred_probs, dim=-1)
        #mask = (pred_labels == init_labels).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        mask = (init_probs[indices,source_classes] - pred_probs[indices,source_classes] < 0.05).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        if ~ (mask.any()):
            print("Iterations stopped at:", i)
            break
        inp = (inp*(~mask) + new_inp*mask).detach()
        inp.requires_grad_()
    pred = model((inp).clamp(0,1))
    probs = softmax(pred)
    final_labels = torch.argmax(probs, dim=-1)
#     print(inp.expand(-1,3,-1,-1).shape, target_image.expand(3,-1,-1).shape)
    lpips_dist_target = lpips_dist(2*inp.expand(-1,3,-1,-1)-1 , 2*target_image.expand(3,-1,-1)-1)
    lpips_dist_target[irrel_images] = -1
#     lpips_dist_source = lpips_dist(2*source_image.expand(-1,3,-1,-1)-1 , 2*inp.expand(-1,3,-1,-1)-1)
#     lpips_dist_source[irrel_images] = -1
    #assert torch.abs(final_labels.squeeze() - init_labels.squeeze()).max() == 0
    return inp, lpips_dist_target, all_confs if get_widths else None

def get_nullspace_projection(J, v):
    y_hat = torch.sum(J*v, -1, keepdim=True)/torch.sum(J * J, -1,  keepdim=True)
    mask = (y_hat <= 0).int()
    x = v - (J * y_hat)*mask 
    return x


if args.dist == 'lpips_alex':
    lpips_dist_func = lpips.LPIPS(net='alex') # best forward scores
else:
    lpips_dist_func = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = args.dataset+"_"+args.model
if False and args.model == 'normal' and args.dataset == 'cifar10':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = resnet.__dict__['resnet20']()
    model.to(device)
    state_dict = torch.load('../pytorch_resnet_cifar10/pretrained_models/resnet20-12fca82f.th')['state_dict']
    new_state_dict = dict([])
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v
    model.load_state_dict(new_state_dict)
    #model.load_state_dict(torch.load('../pytorch_resnet_cifar10/pretrained_models/resnet20-12fca82f.th')['state_dict'])
    model = torch.nn.Sequential(normalize, model)
else:
    model = common.state.State.load('examples/readme/'+model_name+'/classifier.pth.tar').model
    model.to(device)
if args.use_logit_init:
    model_name = model_name+"_loginit"+args.desc

if args.model == 'normal':
    target_layer = model.block2[-1]#model[-1].layer3[-1]
elif args.model == 'at':
    target_layer = model.block2[-1]
elif args.model == 'ccat':
    target_layer = model.block2[-1]
elif args.model == 'msd':
    if args.dataset == 'mnist':
        target_layer = getattr(model, '7')
    else:
        target_layer = model.layer4[-1]
    
lpips_dist_func.to(device)
softmax = torch.nn.Softmax(dim=-1)
for p in model.parameters():
    p.requires_grad_(False)
model.eval()


def lpips_dist(a, b):
    if a.shape[-1] < 32:
        pad_len = (32 - a.shape[-1])//2 + 1	
    else:
        pad_len = 0
    a = torch.nn.functional.pad(a, (pad_len, pad_len, pad_len, pad_len), value=-1)
    b = torch.nn.functional.pad(b, (pad_len, pad_len, pad_len, pad_len), value=-1)
    return lpips_dist_func(a, b)

# train_dataset = torch.utils.data.DataLoader(common.datasets.Cifar10TrainSet(), batch_size=batch_size, shuffle=False, num_workers=0)
if args.dataset == 'cifar10':
    raw_test_dataset = common.datasets.Cifar10TestSet()
    test_dataset = torch.utils.data.DataLoader(common.datasets.Cifar10TestSet(), batch_size=args.batch_size, shuffle=False, num_workers=0)
    classes = ['airplane',
                'automobile',
                'bird',
                'cat',
                'deer',
                'dog',
                'frog',
                'horse',
                'ship',
                'truck']
elif args.dataset == 'mnist':
    raw_test_dataset = common.datasets.MNISTTestSet()
    test_dataset = torch.utils.data.DataLoader(common.datasets.MNISTTestSet(), batch_size=args.batch_size, shuffle=False, num_workers=0)
    classes = [str(i) for i in range(10)]
    
data_dict = dict([(i, list()) for i in range(len(classes))])
try:
    for img, label in raw_test_dataset:
        data_dict[int(label)].append(torch.Tensor(img))
except AssertionError:
    pass

dataset_per_class = [torch.stack(data_dict[i]) for i in range(len(classes))]

def get_targets(start_ind=0, end_ind=10, indices=None, filt=lambda pred_label, i, max_prob: pred_label == i):
    target_img_indices = []
    target_images = []
    if indices:
        target_img_indices = indices
    for i in range(start_ind, end_ind):
        if indices:
            target_images.append(dataset_per_class[i][indices[i]])
            continue
        while True:
            ind = random.randint(0, len(dataset_per_class[i]))
            source_image = common.torch.as_variable(dataset_per_class[i][[ind]], False)
            source_image = source_image.permute(0, 3, 1, 2).to(device)
            pred = model((source_image).clamp(0,1))
            probs = torch.nn.functional.softmax(pred)
            pred_label = torch.argmax(probs, dim=-1)[0]
            max_prob = torch.max(probs, dim=-1)[0]
            print(i, ind, pred_label, max_prob)
            if filt(pred_label, i, max_prob):
                print(i, pred_label)
                target_img_indices.append(ind)
                target_images.append(dataset_per_class[i][ind])
                break
    return target_img_indices, target_images


def run_attack(dataset , target_images, get_widths=False, use_loginit=False, get_images=False):
    inp_per_class = []
    dist_per_class = []
    confs_per_class = []
#     tracemalloc.start()
    for i, image in enumerate(target_images):
        print(i)
        all_inp = []
        all_dist = []
        all_confs = []
        for b, data_batch in enumerate(tqdm(dataset)):
            #print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3)
#             snapshot = tracemalloc.take_snapshot()
#             display_top(snapshot)
            if use_loginit:
                source_image = common.torch.as_variable(data_batch[0], True)
                source_image = source_image.permute(0, 3, 1, 2)
#                 inp1, dist, confs1 = batch_nullspace_attack(data_batch[0], data_batch[1], model, args.step_size, image, args.iters, device, get_widths=get_widths, inp=source_image)
                inp1 = source_image
                #inp2 = inp1 + 10/255*torch.randn(*inp1.shape).to(inp1.device)
                target_img = image.permute(2, 0, 1).repeat(len(inp1), 1, 1, 1).cuda()
                inp2, pert = logit_init(model, inp1, data_batch[1], target_img=target_img)
                pert[i] = inp1[i] - inp2[i]
                inp2 = inp2 + pert
                inp2 = inp2.clone().detach().requires_grad_(True)
                inp3, dist, confs3 = batch_nullspace_attack(inp2, data_batch[1], model, args.step_size, image, args.iters, device, get_widths=get_widths, inp=inp2)
                inp = torch.stack([inp1.cpu().detach(), inp2.cpu().detach(), inp3.cpu().detach()], dim=0)
                confs = []
            else:
                inp, dist, confs = batch_nullspace_attack(data_batch[0], data_batch[1], model, args.step_size, image, args.iters, device, get_widths=get_widths)
                inp = inp.cpu().detach()
                dist = dist.cpu().detach()
#             all_confs.append(confs)
            all_inp.append(inp)
            all_dist.append(dist)
            progress(b, len(dataset) , epoch=i)
        all_inp = torch.cat(all_inp, dim=0)
        all_dist = torch.cat(all_dist, dim=0)
        inp_per_class.append(all_inp)
        dist_per_class.append(all_dist)
        confs_per_class.append(all_confs)
    inp_per_class = torch.stack(inp_per_class, dim=0)
    dist_per_class = torch.stack(dist_per_class, dim=0).cpu().detach().numpy()
    return inp_per_class, dist_per_class, confs_per_class


if args.get_widths:
    incorr_filt = lambda pred_label, i, max_prob: pred_label == 1 - i
    corr_filt = lambda pred_label, i, max_prob: pred_label == i
    start_ind, end_ind = 2, 4
    target_img_indices, target_images = get_targets(start_ind, end_ind, corr_filt)
#     incorr_img_indices, incorr_images = get_targets(2, incorr_filt)
#     example_dataset = [(img.unsqueeze(0), torch.LongTensor([1 - i])) for img, i in zip(incorr_images, range(len(classes)))]
    example_dataset = [(img.unsqueeze(0), torch.LongTensor([i])) for img, i in zip(target_images, range(start_ind, end_ind))]
    inp_per_class, dist_per_class, confs_per_class = run_attack(example_dataset, target_images, get_widths=True) 
    with open('./'+model_name+f'_path_widths_{start_ind}_{end_ind}.pkl', 'wb+') as fp:
        pickle.dump(confs_per_class, fp)
    quit()

def plot_images(inp_per_class, classes, suffix=""):
    cam = AblationCAM(model=model, target_layers=[target_layer], use_cuda=True)
    for args.show_grad in [False]:
        f, axarr = plt.subplots(len(classes),len(classes), figsize=(2*len(classes),2*len(classes))) 
        f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.4)
        for i in range(len(classes)):
            for j in range(len(classes)):
                if j == 0:
                    axarr[i, j].set_ylabel(classes[i])
                if i == 0:
                    axarr[i, j].set_title(classes[j])

                if not args.show_grad:
                    axarr[i, j].imshow(inp_per_class[i, j].numpy().transpose((1, 2, 0))) 
                else:
                    grayscale_cam = cam(input_tensor=inp_per_class[i, j].unsqueeze(0).requires_grad_(), target_category=j)
                    grayscale_cam = grayscale_cam[0, :]
                    visualization = show_cam_on_image(inp_per_class[i, j].clamp(0,1).numpy().transpose((1, 2, 0)), grayscale_cam, use_rgb=True)
                    axarr[i, j].imshow(visualization) 
                axarr[i, j].set_yticklabels([])
                axarr[i, j].set_xticklabels([])
                #axarr[i, j].set_xlabel(np.round(dist_per_class[i, j].squeeze(), 3))
                axarr[i, j].set_xlabel(np.round(float(softmax(model(inp_per_class[i, j].unsqueeze(0).to(device)))[0][j]), 3) )
                if i == j:
                    axarr[i, j].patch.set_edgecolor('black')  
                    axarr[i, j].patch.set_linewidth('5')  
        if not args.show_grad:
            plt.savefig('./'+model_name+suffix+'_examples.pdf')
        else:
            plt.savefig('./'+model_name+suffix+'_examples_gradcam.pdf')

target_img_indices, target_images = get_targets(indices=[474, 385, 884, 133, 381, 408, 641, 613, 384, 240])

if args.examples:
    print(target_img_indices)
    example_dataset = [(torch.stack(target_images, dim=0),  torch.arange(len(classes)))]
    inp_per_class, dist_per_class, confs_per_class = run_attack(example_dataset, target_images, get_widths=False, use_loginit=args.use_logit_init)
    if args.use_logit_init:
        print(inp_per_class.shape)
        inp_per_class_1, inp_per_class_2, inp_per_class_3 = inp_per_class[:,0], inp_per_class[:,1], inp_per_class[:,2]
        plot_images(inp_per_class_1, classes, "_1")
        plot_images(inp_per_class_2, classes, "_2")
        plot_images(inp_per_class_3, classes, "_3")
    else:
        plot_images(inp_per_class, classes, "_1")
    
if args.full_eval:
    print(target_img_indices)
    inp_per_class, dist_per_class, confs_per_class = run_attack(test_dataset, target_images, get_images=False)
    with open('./'+model_name+'_dist.pkl', 'wb+') as fp:
        pickle.dump(dist_per_class, fp)
