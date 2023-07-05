# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import sys
import librosa.display 
import cv2
from matplotlib.pyplot import MultipleLocator # 这个模块用于设置刻度间隔

os.chdir(sys.path[0])
folder = 'sing_data\\美声x民族\\val\\美声'
folder1 = 'sing_data\\美声x民族\\val\\民族'
# folder = 'mei_min_heatmap'

mel_scale = librosa.mel_frequencies(n_mels=256)
print(mel_scale)
freq_male = np.zeros(256)
freq_female = np.zeros(256)

nan = os.listdir(folder)
nv = os.listdir(folder1)

for i in range(len(nan)):
    # cv2 imread() 不接受中文路径
    temp = np.fromfile(os.path.join(folder, nan[i]), dtype=np.uint8)
    img = cv2.imdecode(temp, -1)

    for j in range(256):
        freq_male[j] += np.sum(img[j, :, :])


for i in range(len(nv)):
    # cv2 imread() 不接受中文路径
    temp = np.fromfile(os.path.join(folder1, nv[i]), dtype=np.uint8)
    img = cv2.imdecode(temp, -1)

    for j in range(256):
        freq_female[j] += np.sum(img[j, :, :]) 

# mel_scale = (mel_scale/100)
# female = female.astype(np.int64)
# female = female / np.max(female)

for i in range(256):
    print('mel', mel_scale[i])
    print(freq_male[i])


# plt.figure(figsize=(10,7))
# plt.bar(mel_scale, ffreq_male, width=0.5, color = 'blue', edgecolor = 'blue', linewidth = 0.2)
# plt.xlabel('Frequency / *100Hz')
# plt.ylabel('Amp Sum')
# plt.xlim(0, 110)
# plt.title('Male amp sum')
# x_major_locator=MultipleLocator(5)
# ax = plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)

# plt.show()

# folder2 = 'male_female_heatmap'
folder2 = 'mei_min_heatmap'

male = np.zeros(256)
female = np.zeros(256)
f1 = os.listdir(folder2)
for i in range(len(f1)):
    img = cv2.imread(os.path.join(folder2, f1[i]))

    if int(((f1[i])[-5:])[0]) == 0:
        # 统计直方图 
        # print(123)
        for j in range(256):
            # print(np.sum(img[j, :, :]))
            female[j] += np.sum(img[j, :, :])
    elif int(((f1[i])[-5:])[0]) == 1:
        # print(234)
        for j in range(256):
            male[j] += np.sum(img[j, :, :])

mel_scale = (mel_scale/100)

male = male / np.max(male)
female = female / np.max(female)

freq_male = freq_male/ np.max(freq_male)
freq_female = freq_female / np.max(freq_female)

plt.figure(figsize=(10,7))
plt.plot(mel_scale, female, color = 'red', linewidth = 2, label='minzu heat')
plt.plot(mel_scale, male, color = 'blue', linewidth = 2, label='meisheng heat')
plt.plot(mel_scale, freq_male, color = 'black', linewidth = 1, label='meisheng freq amp')
plt.plot(mel_scale, freq_female, color = 'purple', linewidth = 1, label='minzu freq amp')
plt.xlabel('Frequency / *100Hz')
plt.ylabel('Heat')
plt.xlim(0, 110)
plt.legend(loc='upper right')
x_major_locator=MultipleLocator(5)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.show()