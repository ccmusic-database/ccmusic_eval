import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def draw_CAM(model, val_images, save_path, c, visual_heatmap=False):
    '''
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''

    # 获取模型输出的feature/score
    # 利用原始的.pth模型进行forward之前，一定要先进行model.eval()，不启用BN和DP
    model.eval() # 不启用batchNormalization和Dropout。
    features = model.conv_model(val_images)
    features1 = model.spp_layer(features)
    output = model.linear_model(features1)
    # output = model(img)

    def extract(grad):
        global features_grad
        features_grad = grad

    # 预测得分最高的一类对应的输出score

    # index = np.argmax(output.cpu().data.numpy())
    # print(index)
    # index = index[np.newaxis, np.newaxis]
    # index = torch.from_numpy(index)
    # one_hot = torch.zeros(1,2).scatter_(1, index, 1)
    # one_hot.requires_grad = True
    # pred_class = torch.sum(one_hot * output)
    # pred_class.backward() # 计算梯度

    # print(index)
    # print(pred_class)
    cc = 0
    index = []
    for i in range(output.shape[0]):
        index.append(torch.argmax(output[i]).item())
    in1 = copy.deepcopy(index)
    print(in1)

    index = np.array(index)
    index = np.int64(torch.from_numpy(index[np.newaxis, np.newaxis]))
 

    # pred_class = torch.zeros(output.shape[0])
    # for i in range(output.shape[0]):
    #     pred_class[i] = torch.max(output[i])    
    # pred_class = torch.sum(pred_class)
    # print(pred_class)
    pred_class = torch.zeros(output.shape[0])
    for i in range(output.shape[0]):
        pred_class[i] = torch.sum(output[i])
    pred_class = torch.sum(pred_class)
    print(pred_class)
 
    features.register_hook(extract)

    # for i in range(len(output)):
    # print('output[i]', output[i])
        # pred = torch.argmax(output[i]).item() # 得到16个

        # # print(output, pred)
        # pred_class = output[i, pred]
        # print(type(pred_class))
        # print(pred_class)
    
    pred_class.backward(retain_graph=True) # 计算梯度

    grads = features_grad # 获取梯度
    # print(grads.shape)
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1,1))
    # print(grads.shape, grads)
    # print(pooled_grads.shape)

    # 此处batch size默认为1， 所以去掉了第0维度 （batch size维）
    for ii in range(output.shape[0]):
        p_grads = pooled_grads[ii]

        
        # for i in features[0]:
        #     f[i, :, :, :] = features[]
        f = (features[ii])
        # features = features[0]
        # print(f.shape, pooled_grads.shape)
        var_channel_last_layer = 128

        # 最后一层visualize
        for j in range(var_channel_last_layer):
            f[j, ...] *= p_grads[j, ...]



        '''----------------------------- '''
        heatmap = f.cpu().detach().numpy()
        heatmap = np.mean(heatmap, axis=0)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        img = val_images.cpu().numpy()[i,:,:,:]
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[2]))

        # 可视化原始热力图
        # if visual_heatmap:
        #     plt.matshow(heatmap)
        #     plt.show()


        # print('heat shape',heatmap.shape)
        # print('img shape', img.shape)
        # print(img)
        heatmap = np.uint8(255*heatmap) # 将热力图转换为RGB格式
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # 将热力图应用于原始图像
        heatmap = np.float32(heatmap) / 255
        

        image = np.zeros([256, 256, 3])
        for jj in range(3):
            image[:,:,jj] =  img[jj,:,:]

        image -= np.min(image)
        image /= np.max(image)

        # plt.matshow(image)
        # plt.show()
        cam = heatmap  + image # 0.4 是热力图的强度因子
        cam = cam /np.max(cam)
        # plt.matshow(cam)
        # plt.show()
        name = str(str(c) + '_' + str(cc) + '_' + str(in1[ii]))
 
        cv2.imwrite(save_path +'/' + name + '.png', cam*255) 
        cc += 1