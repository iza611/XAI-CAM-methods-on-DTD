import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from glob import glob
import imageio
import torch.backends.cudnn as cudnn
from modules.resnet import resnet50, resnet101, resnet18
import matplotlib.cm
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import cv2
from dtd_index import index2class
from LRP_util import *
import os
import argparse

def get_CAM(method, layer, input, target, idx=1000, save=False): # DATA VARIABLE HAS TO BE ADDED TO RUN RUN.IPYNB!!!!!

    """ 
    method can be either Grad_CAM, Grad_CAMpp, Score_CAM, or Relevance_CAM 
    layer can be equal to 2 or 4
    """

    model = resnet50(pretrained=True).eval()
    if(layer==2):
        args_target_layer = 'layer2'
        target_layer = model.layer2
    elif(layer==4):
        args_target_layer = 'layer4'
        target_layer = model.layer4
    else:
        return 'wrong layer. set to 2 or 4.'
    args_target_class = None

    value = dict()
    def forward_hook(module, input, output):
        value['activations'] = output
    def backward_hook(module, input, output):
        value['gradients'] = output[0]

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    Score_CAM_class = ScoreCAM(model,target_layer)

    image = input
    label = target
    np_image = image.transpose((1, 2, 0)).copy()
    image = torch.from_numpy(image)
    image = image.reshape(1, 3, 224, 224)
    img_show = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    img_show = cv2.resize(img_show,(224,224))

    # image, label = data
    # print(image.shape)
    # label = label[0].item()
    # np_image = torch.permute(image[0], (1, 2, 0)).numpy()
    # print(np_image.shape)
    # img_show = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    # img_show = cv2.resize(img_show,(224,224))

    R_CAM, output = model(image, args_target_layer, [args_target_class])

    if args_target_class == None:
        maxindex = np.argmax(output.data.cpu().numpy())
    else:
        maxindex = args_target_class

    if(save):

        save_dir_fig1 = f'data/R_CAM_results_layer{layer}/figs/heatmaps'
        save_dir_fig2 = f'data/R_CAM_results_layer{layer}/figs/segmented_imgs'
        save_dir_np1 = f'data/R_CAM_results_layer{layer}/np/heatmaps'
        save_dir_np2 = f'data/R_CAM_results_layer{layer}/np/segmented_imgs'
        save_name = '/{}_{}'.format(index2class[maxindex][:10], index2class[label] + str(idx))

    output[:, maxindex].sum().backward(retain_graph=True)
    activation = value['activations']  # [1, 2048, 7, 7]
    gradient = value['gradients']  # [1, 2048, 7, 7]
    gradient_2 = gradient ** 2
    gradient_3 = gradient ** 3

    if(method=='Grad_CAM'):
        
        gradient_ = torch.mean(gradient, dim=(2, 3), keepdim=True)
        grad_cam = activation * gradient_
        grad_cam = torch.sum(grad_cam, dim=(0, 1))
        grad_cam = torch.clamp(grad_cam, min=0)
        grad_cam = grad_cam.data.cpu().numpy()
        grad_cam = cv2.resize(grad_cam, (224, 224))
        grad_cam_s = img_show*threshold(grad_cam)[...,np.newaxis]

        if(save):

            print("grad_cam finished.\nsaving...")

            plt.imshow((grad_cam),cmap='seismic')
            plt.imshow(img_show, alpha=.5)
            plt.title('Grad CAM', fontsize=15)
            plt.axis('off')
            plt.savefig(save_dir_fig1 + '/Grad_CAM' + save_name)
            plt.clf()
            plt.close()

            np.save(save_dir_np1 + '/Grad_CAM' + save_name, grad_cam)

            plt.imshow(grad_cam_s)
            plt.title('Grad CAM', fontsize=15)
            plt.axis('off')
            plt.savefig(save_dir_fig2 + '/Grad_CAM' + save_name)
            plt.clf()
            plt.close()

            np.save(save_dir_np2 + '/Grad_CAM' + save_name, grad_cam_s)
            print("saved.")

        return grad_cam

    if(method=='Grad_CAMpp'):

        alpha_numer = gradient_2
        alpha_denom = 2 * gradient_2 + torch.sum(activation * gradient_3, axis=(2, 3), keepdims=True)  # + 1e-2
        alpha = alpha_numer / alpha_denom
        w = torch.sum(alpha * torch.clamp(gradient, 0), axis=(2, 3), keepdims=True)
        grad_campp = activation * w
        grad_campp = torch.sum(grad_campp, dim=(0, 1))
        grad_campp = torch.clamp(grad_campp, min=0)
        grad_campp = grad_campp.data.cpu().numpy()
        grad_campp = cv2.resize(grad_campp, (224, 224))
        grad_campp_s = img_show*threshold(grad_campp)[...,np.newaxis]

        if(save):
            print("grad_campp finished.\nsaving...")

            plt.imshow((grad_campp),cmap='seismic')
            plt.imshow(img_show, alpha=.5)
            plt.title('Grad CAM++', fontsize=15)
            plt.axis('off')
            plt.savefig(save_dir_fig1 + '/Grad_CAM++' + save_name)
            plt.clf()
            plt.close()

            np.save(save_dir_np1 + '/Grad_CAM++' + save_name, grad_campp)

            plt.imshow(grad_campp_s)
            plt.title('Grad CAM++', fontsize=15)
            plt.axis('off')
            plt.savefig(save_dir_fig2 + '/Grad_CAM++' + save_name)                                                                                                      
            plt.clf()
            plt.close()

            np.save(save_dir_np2 + '/Grad_CAM++' + save_name, grad_campp_s)
            print("saved.")

        return grad_campp

    if(method=='Score_CAM'):

        score_map, _ = Score_CAM_class(image, class_idx=maxindex)
        score_map = score_map.squeeze()
        score_map = score_map.detach().cpu().numpy()
        score_map_s = img_show*threshold(score_map)[...,np.newaxis]

        if(save):

            print("score_map finished.\nsaving...")

            plt.imshow((score_map),cmap='seismic')
            plt.imshow(img_show, alpha=.5)
            plt.title('Score_CAM', fontsize=15)
            plt.axis('off')
            plt.savefig(save_dir_fig1 + '/Score_CAM' + save_name)
            plt.clf()
            plt.close()

            np.save(save_dir_np1 + '/Score_CAM' + save_name, score_map)

            plt.imshow(score_map_s)
            plt.title('Score_CAM', fontsize=15)
            plt.axis('off')
            plt.savefig(save_dir_fig2 + '/Score_CAM' + save_name)
            plt.clf()
            plt.close()

            np.save(save_dir_np2 + '/Score_CAM' + save_name, score_map_s)
            print("saved.")

        return score_map

    if(method=='Relevance_CAM'):

        R_CAM = tensor2image(R_CAM)
        R_CAM_s = img_show*threshold(R_CAM)[...,np.newaxis]

        if(save):

            print("R_CAM finished.\nsaving...")

            plt.imshow((R_CAM),cmap='seismic')
            plt.imshow(img_show, alpha=.5)
            plt.title('Relevance_CAM', fontsize=15)
            plt.axis('off')
            plt.savefig(save_dir_fig1 + '/Relevance_CAM' + save_name)
            plt.clf()
            plt.close()

            np.save(save_dir_np1 + '/Relevance_CAM' + save_name, R_CAM)

            plt.imshow(R_CAM_s)
            plt.title('Relevance_CAM', fontsize=15)
            plt.axis('off')
            plt.savefig(save_dir_fig2 + '/Relevance_CAM' + save_name)
            plt.clf()
            plt.close()

            np.save(save_dir_np2 + '/Relevance_CAM' + save_name, R_CAM_s)
            print("saved.")

        return R_CAM

    return 3

# idx = 0
# for data in dataloader:
#     image, label = data
#     label = label[0].item()
#     np_image = torch.permute(image[0], (1, 2, 0)).numpy()
#     img_show = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
#     img_show = cv2.resize(img_show,(224,224))

#     R_CAM, output = model(image, args_target_layer, [args_target_class])

#     if args_target_class == None:
#         maxindex = np.argmax(output.data.cpu().numpy())
#     else:
#         maxindex = args_target_class

#     print(index2class[maxindex])
#     save_dir_fig1 = '../data/R_CAM_results/figs/heatmaps'
#     save_dir_fig2 = '../data/R_CAM_results/figs/segmented_imgs'
#     save_dir_np1 = '../data/R_CAM_results/np/heatmaps'
#     save_dir_np2 = '../data/R_CAM_results/np/segmented_imgs'
#     save_name = '/{}_{}'.format(index2class[maxindex][:10], index2class[label] + str(idx))

#     output[:, maxindex].sum().backward(retain_graph=True)
#     activation = value['activations']  # [1, 2048, 7, 7]
#     gradient = value['gradients']  # [1, 2048, 7, 7]
#     gradient_2 = gradient ** 2
#     gradient_3 = gradient ** 3

#     gradient_ = torch.mean(gradient, dim=(2, 3), keepdim=True)
#     grad_cam = activation * gradient_
#     grad_cam = torch.sum(grad_cam, dim=(0, 1))
#     grad_cam = torch.clamp(grad_cam, min=0)
#     grad_cam = grad_cam.data.cpu().numpy()
#     grad_cam = cv2.resize(grad_cam, (224, 224))
#     grad_cam_s = img_show*threshold(grad_cam)[...,np.newaxis]

#     print("grad_cam finished.\nsaving...")

#     plt.imshow((grad_cam),cmap='seismic')
#     plt.imshow(img_show, alpha=.5)
#     plt.title('Grad CAM', fontsize=15)
#     plt.axis('off')
#     plt.savefig(save_dir_fig1 + '/Grad_CAM' + save_name)
#     plt.clf()
#     plt.close()

#     np.save(save_dir_np1 + '/Grad_CAM' + save_name, grad_cam)

#     plt.imshow(grad_cam_s)
#     plt.title('Grad CAM', fontsize=15)
#     plt.axis('off')
#     plt.savefig(save_dir_fig2 + '/Grad_CAM' + save_name)
#     plt.clf()
#     plt.close()

#     np.save(save_dir_np2 + '/Grad_CAM' + save_name, grad_cam_s)
#     print("saved.")

#     alpha_numer = gradient_2
#     alpha_denom = 2 * gradient_2 + torch.sum(activation * gradient_3, axis=(2, 3), keepdims=True)  # + 1e-2
#     alpha = alpha_numer / alpha_denom
#     w = torch.sum(alpha * torch.clamp(gradient, 0), axis=(2, 3), keepdims=True)
#     grad_campp = activation * w
#     grad_campp = torch.sum(grad_campp, dim=(0, 1))
#     grad_campp = torch.clamp(grad_campp, min=0)
#     grad_campp = grad_campp.data.cpu().numpy()
#     grad_campp = cv2.resize(grad_campp, (224, 224))
#     grad_campp_s = img_show*threshold(grad_campp)[...,np.newaxis]

#     print("grad_campp finished.\nsaving...")

#     plt.imshow((grad_campp),cmap='seismic')
#     plt.imshow(img_show, alpha=.5)
#     plt.title('Grad CAM++', fontsize=15)
#     plt.axis('off')
#     plt.savefig(save_dir_fig1 + '/Grad_CAM++' + save_name)
#     plt.clf()
#     plt.close()

#     np.save(save_dir_np1 + '/Grad_CAM++' + save_name, grad_campp)

#     plt.imshow(grad_campp_s)
#     plt.title('Grad CAM++', fontsize=15)
#     plt.axis('off')
#     plt.savefig(save_dir_fig2 + '/Grad_CAM++' + save_name)                                                                                                      
#     plt.clf()
#     plt.close()

#     np.save(save_dir_np2 + '/Grad_CAM++' + save_name, grad_campp_s)
#     print("saved.")

#     score_map, _ = Score_CAM_class(image, class_idx=maxindex)
#     score_map = score_map.squeeze()
#     score_map = score_map.detach().cpu().numpy()
#     score_map_s = img_show*threshold(score_map)[...,np.newaxis]

#     print("score_map finished.\nsaving...")

#     plt.imshow((score_map),cmap='seismic')
#     plt.imshow(img_show, alpha=.5)
#     plt.title('Score_CAM', fontsize=15)
#     plt.axis('off')
#     plt.savefig(save_dir_fig1 + '/Score_CAM' + save_name)
#     plt.clf()
#     plt.close()

#     np.save(save_dir_np1 + '/Score_CAM' + save_name, score_map)

#     plt.imshow(score_map_s)
#     plt.title('Score_CAM', fontsize=15)
#     plt.axis('off')
#     plt.savefig(save_dir_fig2 + '/Score_CAM' + save_name)
#     plt.clf()
#     plt.close()

#     np.save(save_dir_np2 + '/Score_CAM' + save_name, score_map_s)
#     print("saved.")

#     R_CAM = tensor2image(R_CAM)
#     R_CAM_s = img_show*threshold(R_CAM)[...,np.newaxis]

#     print("R_CAM finished.\nsaving...")

#     plt.imshow((R_CAM),cmap='seismic')
#     plt.imshow(img_show, alpha=.5)
#     plt.title('Relevance_CAM', fontsize=15)
#     plt.axis('off')
#     plt.savefig(save_dir_fig1 + '/Relevance_CAM' + save_name)
#     plt.clf()
#     plt.close()

#     np.save(save_dir_np1 + '/Relevance_CAM' + save_name, R_CAM)

#     plt.imshow(R_CAM_s)
#     plt.title('Relevance_CAM', fontsize=15)
#     plt.axis('off')
#     plt.savefig(save_dir_fig2 + '/Relevance_CAM' + save_name)
#     plt.clf()
#     plt.close()

#     np.save(save_dir_np2 + '/Relevance_CAM' + save_name, R_CAM_s)
#     print("saved.")

#     idx += 1

# print('Done')