# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import sys
import librosa.display 
import cv2
import colorsys

from matplotlib.pyplot import MultipleLocator, gray # 这个模块用于设置刻度间隔

os.chdir(sys.path[0])
folder = 'male_female_heatmap'
# folder = 'mei_min_heatmap'

mel_scale = librosa.mel_frequencies(n_mels=256)

male = np.zeros(256)
female = np.zeros(256)

f1 = os.listdir(folder)
c1 = 0
c2 = 0

for i in range(len(f1)):
    # print(f1[i])
    i1 = cv2.imread(os.path.join(folder, f1[i])) # [height, width, channel]
    # i2 = cv2.cvtColor(i1, cv2.COLOR_BGR2LUV) # RBG先转LUV再转gray
    # # plt.imshow(img)
    # # plt.show()
    # # os.system('pause')
    # # 转gray
    # img = np.zeros((i2.shape[0], i2.shape[1]))
    # for y_pos in range(i2.shape[0]):
    #     for x_pos in range(i2.shape[1]):
    #         color = i2[y_pos, x_pos]
    #         r,g,b = color 
    #         h, _, _ = colorsys.rgb_to_hls(r, g, b)
    #         img[y_pos, x_pos] = 1.0 - h
    
    gray_values = np.arange(256, dtype=np.uint8)
    print(gray_values.shape)
    print(cv2.applyColorMap(gray_values, cv2.COLORMAP_JET).shape)
    
    color_values = map(tuple, cv2.applyColorMap(gray_values, cv2.COLORMAP_JET).reshape(256, 3))
    
    color_to_gray_map = dict(zip(color_values, gray_values))
    print(type(color_to_gray_map))
    img = np.apply_along_axis(lambda bgr: color_to_gray_map[tuple(bgr)], 2, i1)
    # print(img.shape)
    # plt.imshow(img, cmap='gray')
    # plt.show()
    # plt.imshow(i1)
    # plt.show()
    # os.system('pause')
    img = img[::-1,:]
    if int(((f1[i])[-5:])[0]) == 0:
        # 统计直方图 
        c1 += 1

        for j in range(256):
            # print(np.sum(img[j, :, :]))
            # female[j] += np.sum(img[j, :, :])
            female[j] += np.sum(img[j, :])
    elif int(((f1[i])[-5:])[0]) == 1:
        c2 += 1
        for j in range(256):
            # male[j] += np.sum(img[j, :, :])
            male[j] += np.sum(img[j, :])
print(c1, c2)
# mel_scale = (mel_scale/100).astype(np.int64)
# print(mel_scale)
# os.system('pause')

mel_scale = (mel_scale/100)
female = female.astype(np.int64)
female = female / np.max(female)
male = male.astype(np.int64)
male = male / np.max(male)
x_major_locator=MultipleLocator(5)

# plt.figure(figsize=(10,7))
# plt.bar(mel_scale, female, width=0.5, color = 'red', edgecolor = 'red', linewidth = 0.2)
# plt.xlabel('Frequency / *100Hz')
# plt.ylabel('Heat')
# plt.xlim(0, 110)
# plt.title('female')
# # plt.title('minzu')
# x_major_locator=MultipleLocator(5)
# ax = plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# # norm = plt.Normalize(-1, 1)
# # norm_values = norm(key_values)
# # map_bir = cm.get_camp(name='inferno')
# # sm = cm.ScalarMappable(camp=map_vir, norm=norm)
# # plt.title(u'直方图')
# plt.show()

# plt.figure(figsize=(10,7))
# male = male.astype(np.int64)
# male = male / np.max(male)
# plt.bar(mel_scale, male, width=0.5, color='blue', edgecolor = 'blue', linewidth = 0.2)
# plt.xlabel('Frequency / *100Hz')
# plt.ylabel('Heat')
# plt.xlim(0, 110)
# plt.title('male')
# # plt.title('meisheng')
# ax = plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# # norm = plt.Normalize(-1, 1)
# # norm_values = norm(key_values)
# # map_bir = cm.get_camp(name='inferno')
# # sm = cm.ScalarMappable(camp=map_vir, norm=norm)
# # plt.title(u'直方图')
# plt.show()

plt.figure(figsize=(10,7))
plt.plot(mel_scale, female, linewidth=2, color = 'red', label='female')
plt.plot(mel_scale, male, linewidth=2, color = 'blue', label='male')
plt.xlabel('Frequency / *100Hz')
plt.ylabel('Heat')
plt.xlim(0, 110)
plt.legend(loc='upper right')
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.title('Male Female heat map sum')
plt.show()
# os.system('pause')

# fig.colorbar(img, ax=ax, format='%+2.0f dB')

# ax.set(title='Mel-frequency spectrogram')