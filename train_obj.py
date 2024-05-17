import os
import wandb
import datetime
import numpy as np
import os.path as osp
from easydict import EasyDict

from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from models import LatentEBM, LatentEBM128
from new_dataset import Clevr, ClevrTex, ClevrStyle


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--entity', type=str, default='movinghoon')
    parser.add_argument('--seed', default=42, type=int)
    
    parser.add_argument('--n_slots', default=4, type=int, help='number of components to explain an image with')
    parser.add_argument('--dataset', default='clevr_easy', type=str, help='Dataset to use (intphys or others or imagenet or cubes)')
    
    parser.add_argument('--batch_size', default=24, type=int, help='size of batch of input to use')
    
    parser.add_argument('--img_size', default=64, type=int)
    
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate for training')
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--log_interval', default=1000, type=int)
    parser.add_argument('--save_interval', default=5000, type=int)
    
    parser.add_argument('--filter_dim', default=64, type=int, help='number of filters to use')
    parser.add_argument('--latent_dim', default=16, type=int, help='dimension of the latent')

    parser.add_argument('--num_steps', default=5, type=int, help='Steps of gradient descent for training')
    parser.add_argument('--num_visuals', default=16, type=int, help='Number of visuals')
    parser.add_argument('--num_additional', default=0, type=int, help='Number of additional components to add')

    parser.add_argument('--step_lr', default=1000.0, type=float, help='step size of latents')
    return parser.parse_args()


def init_model(args, dataset):
    if args.img_size == 64:
        model = LatentEBM(args, dataset).cuda()
    else:
        model = LatentEBM128(args, dataset).cuda()
    models = [model for _ in range(args.ensemble)]
    optimizers = [torch.optim.Adam(model.parameters(), lr=args.lr) for _ in range(args.ensemble)]
    return models, optimizers


# TODO. Data pathes
DATA_PATH = {
    'clevr_easy': '/data1/common_datasets/neural_systematic_binder_dataset/clevr-easy',
    'clevr_hard': '/data1/common_datasets/neural_systematic_binder_dataset/clevr-hard',
    'clevr': '/data1/common_datasets/object_centric',
    'clevrtex': '/data1/common_datasets/object_centric',
}

DSET_DICT = {
    'clevr_easy':ClevrStyle,
    'clevr_hard':ClevrStyle,
    'clevr':Clevr,
    'clevrtex':ClevrTex,
}


def gen_image(latents, args, models, im_neg, im, num_steps, sample=False, create_graph=True):
    im_noise = torch.randn_like(im_neg).detach()
    
    im_negs = []
    latents = torch.stack(latents, dim=0)
    im_neg.requires_grad_(requires_grad=True)
    s = im.size()
    masks = torch.zeros(s[0], args.components, s[-2], s[-1]).to(im_neg.device)
    masks.requires_grad_(requires_grad=True)

    for i in range(num_steps):
        im_noise.normal_()

        energy = 0
        for j in range(len(latents)):
            energy = models[j % args.components].forward(im_neg, latents[j]) + energy

        im_grad, = torch.autograd.grad([energy.sum()], [im_neg], create_graph=create_graph)

        im_neg = im_neg - args.step_lr * im_grad

        latents = latents

        im_neg = torch.clamp(im_neg, 0, 1)
        im_negs.append(im_neg)
        im_neg = im_neg.detach()
        im_neg.requires_grad_()
        
    return im_neg, im_negs, im_grad, masks


def train(models, optimizers, train_loader, epoch, args):
    # set training mode
    models = [model.train() for model in models]    # ?
    iters_per_epoch = len(train_loader)
    
    # for logging
    loss_list, ml_loss_list, energy_pos_list, energy_neg_list = [], [], [], []
    
    # train
    for i, batch in enumerate(tqdm(train_loader, total=len(train_loader))):
        im = batch['image'].cuda()
        
        # extract latent
        latent = models[0].embed_latent(im)
        latents = torch.chunk(latent, args.components, dim=1)
        
        # gen image
        im_neg = torch.rand_like(im)
        im_neg, im_negs, im_grad, _ = gen_image(latents, args, models, im_neg, im, args.num_steps, args.sample)
        im_negs = torch.stack(im_negs, dim=1)
        
        # energy
        energy_pos = 0
        energy_neg = 0
        
        # compute energy
        energy_poss = []
        energy_negs = []
        for i in range(args.components):
            energy_poss.append(models[i].forward(im, latents[i]))
            energy_negs.append(models[i].forward(im_neg.detach(), latents[i]))
        energy_pos = torch.stack(energy_poss, dim=1)
        energy_neg = torch.stack(energy_negs, dim=1)
        
        # ml loss for logging
        ml_loss = (energy_pos - energy_neg).mean()
        
        # im loss
        loss = torch.pow(im_negs[:, -1:] - im[:, None], 2).mean()
        
        # backward
        loss.backward()
        
        # clip + step + zero-grad
        [torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0) for model in models]
        [optimizer.step() for optimizer in optimizers]
        [optimizer.zero_grad() for optimizer in optimizers]
        
        # logging
        loss_list.append(loss.item())
        ml_loss_list.append(ml_loss.item())
        energy_pos_list.append(energy_pos.mean().item())
        energy_neg_list.append(energy_neg.mean().item())
        
        # logging
        num_iters = iters_per_epoch * epoch + (i + 1)
        if num_iters % args.log_interval == 0:
            out_dict = {
                'epoch': epoch,
                'iters': num_iters,
                'loss': np.mean(loss_list),
                'ml_loss': np.mean(ml_loss_list),
                'energy_pos': np.mean(energy_pos_list),
                'energy_neg': np.mean(energy_neg_list),
            }
            wandb.log(out_dict, commit=True)
            
            # clear
            loss_list, ml_loss_list, im_loss_list, energy_pos_list, energy_neg_list = [], [], [], [], []
            
        if num_iters % args.save_interval == 0:
            model_path = osp.join(args.log_dir, "model_{}.pth".format(num_iters))
            save_dict = {
                'epoch': epoch + 1,
                'iters': num_iters,
                'args': args,
            }
            for i in range(len(models)):
                save_dict['model_state_dict_{}'.format(i)] = models[i].state_dict()
                save_dict['optimizer_state_dict_{}'.format(i)] = optimizers[i].state_dict()
            torch.save(save_dict, model_path)


def test(models, test_loader, epoch, args):
    [model.eval() for model in models]
    for i, batch in enumerate(test_loader):
        im = batch['image'].cuda()
        im = im[:args.num_visuals]
        batch_size = im.size(0)
        
        # latent
        latent = models[0].embed_latent(im)
        latents = torch.chunk(latent, args.components, dim=1)
        
        # gen image
        im_init = torch.rand_like(im)
        im_neg, _, im_grad, mask = gen_image(latents, args, models, im_init, im, args.num_steps, sample=args.sample, 
                                       create_graph=False)
        im_neg = im_neg.detach()
        im_components = []
        
        # 
        if args.components > 1:
            for i, latent in enumerate(latents):
                im_init = torch.rand_like(im)
                latents_select = latents[i:i+1]
                im_component, _, _, _ = gen_image(latents_select, args, models, im_init, im, args.num_steps, sample=args.sample,
                                           create_graph=False)
                im_components.append(im_component)

            im_init = torch.rand_like(im)
            latents_perm = [torch.cat([latent[i:], latent[:i]], dim=0) for i, latent in enumerate(latents)]
            im_neg_perm, _, im_grad_perm, _ = gen_image(latents_perm, args, models, im_init, im, args.num_steps, sample=args.sample,
                                                     create_graph=False)
            im_neg_perm = im_neg_perm.detach()
            im_init = torch.rand_like(im)
            add_latents = list(latents)
            for i in range(args.num_additional):
                add_latents.append(torch.roll(latents[i], i + 1, 0))
            im_neg_additional, _, _, _ = gen_image(tuple(add_latents), args, models, im_init, im, args.num_steps, sample=args.sample,
                                                     create_graph=False)
        
        # grad
        im_grads = []
        im.requires_grad = True
        for i, latent in enumerate(latents):
            energy_pos = models[i].forward(im, latents[i])
            im_grad = torch.autograd.grad([energy_pos.sum()], [im])[0]
            im_grads.append(im_grad)
        im_grad = torch.stack(im_grads, dim=1)
        
        # log images
        s = im.size()
        im_size = s[-1]

        im_grad = im_grad.view(batch_size, args.components, 3, im_size, im_size) # [4, 3, 3, 128, 128]
        im_grad_dense = im_grad.view(batch_size, args.components, 1, 3 * im_size * im_size, 1) # [4, 3, 1, 49152, 1]
        im_grad_min = im_grad_dense.min(dim=3, keepdim=True)[0]
        im_grad_max = im_grad_dense.max(dim=3, keepdim=True)[0] # [4, 3, 1, 1, 1]

        im_grad = (im_grad - im_grad_min) / (im_grad_max - im_grad_min + 1e-5) # [4, 3, 3, 128, 128]
        im_grad[:, :, :, :1, :] = 1
        im_grad[:, :, :, -1:, :] = 1
        im_grad[:, :, :, :, :1] = 1
        im_grad[:, :, :, :, -1:] = 1
        im_output = im_grad.permute(0, 3, 1, 4, 2).reshape(batch_size * im_size, args.components * im_size, 3)
        im_output = im_output.cpu().detach().numpy() * 100
        im_output = (im_output - im_output.min()) / (im_output.max() - im_output.min())

        im = im.cpu().detach().numpy().transpose((0, 2, 3, 1)).reshape(batch_size*im_size, im_size, 3)

        im_output = np.concatenate([im_output, im], axis=1)
        im_output = (im_output * 255).astype(np.uint8)
        
        # TODO: image log
        outs = {
            'val/epoch': epoch,
            'val/grad': wandb.Image(im_output)
        }
        
        step = epoch
        if args.components > 1:
            im_neg_perm = im_neg_perm.detach().cpu()
            im_components_perm = []
            for i,im_component in enumerate(im_components):
                im_components_perm.append(torch.cat([im_component[i:], im_component[:i]]).detach().cpu())
            im_neg_perm = torch.cat([im_neg_perm] + im_components_perm)
            im_neg_perm = np.clip(im_neg_perm, 0.0, 1.0)
            im_neg_perm = make_grid(im_neg_perm, nrow=int(im_neg_perm.shape[0] / (args.components + 1))).permute(1, 2, 0)
            im_neg_perm = (im_neg_perm.numpy()*255).astype(np.uint8)
            
            # TODO: image log
            outs['val/gen_perm'] = wandb.Image(im_neg_perm)
            # imwrite("result/%s/s%08d_gen_perm.png" % (args.exp,step), im_neg_perm)

            im_neg_additional = im_neg_additional.detach().cpu()
            for i in range(args.num_additional):
                im_components.append(torch.roll(im_components[i].detach().cpu(), i + 1, 0))
            im_components = [x.detach().cpu() for x in im_components]
            im_neg_additional = torch.cat([im_neg_additional] + im_components)
            im_neg_additional = np.clip(im_neg_additional, 0.0, 1.0)
            im_neg_additional = make_grid(im_neg_additional, 
                                nrow=int(im_neg_additional.shape[0] / (args.components + args.num_additional + 1))).permute(1, 2, 0)
            im_neg_additional = (im_neg_additional.numpy()*255).astype(np.uint8)
            
            # TODO: image log
            # imwrite("result/%s/s%08d_gen_add.png" % (args.exp,step), im_neg_additional)
            outs['val/gen_add'] = wandb.Image(im_neg_additional)

            print('test at step %d done!' % step)
        wandb.log(outs, commit=True)
        break

def main():
    args = parse_args()
    
    # defaults
    args.components = args.n_slots
    args.ensemble = args.components
    args.tie_weight = True
    args.sample = True
    args.recurrent_model = True
    args.pos_embed = True
    
    # log_dir
    args.exp_name = args.dataset + '_' + args.exp_name + '-' + datetime.now().strftime('%m-%d-%H')
    args.log_dir = './experiments/' + args.exp_name + '/'
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    
    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # wandb
    wandb.init(name=args.exp_name, project='comet', entity=args.entity, config=args)

    # dataset
    img_size = (args.img_size, args.img_size)
    dset_class = DSET_DICT[args.dataset]
    dset = dset_class(DATA_PATH[args.dataset], split='train', img_size=img_size)
    test_dset = dset_class(DATA_PATH[args.dataset], split='test', img_size=img_size)

    # init model
    models, optimizers = init_model(args, dset)
    
    # data loader
    train_loader = DataLoader(dset, 
                              num_workers=4,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True)
    test_loader = DataLoader(test_dset, 
                             num_workers=4,
                             batch_size=args.batch_size,
                             shuffle=False,
                             pin_memory=True)
    
    # run
    for epoch in range(args.epochs):
        
        # train
        train(models, optimizers, train_loader, epoch, args)
        
        # test
        test(models, test_loader, epoch, args)
            
    # finish wandb
    wandb.finish()


if __name__ == "__main__":
    main()
