#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os.path
import random

import matplotlib.pyplot as plt
import pandas as pd

plt.rcdefaults()

from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray

# In[2]:


imagepath = "RawData/"
image_path = "RawData/R-233_5-8-6_000110.T000.D000.P000.H000.PLIF1.TIF"

# In[3]:


import cv2


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


x = "RawData/R-233_5-8-6_000110.T000.D000.P000.H000.PLIF1.TIF"
original = cv2.imread(x, 1)
plt.imshow(original)
plt.show()

gamma = 5
adjusted = adjust_gamma(original, gamma=gamma)
plt.imshow(adjusted)
plt.show()


# In[4]:


def test_log(image):
    image_gray = rgb2gray(image)

    blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=0.1)

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=0.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=0.01)

    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ["yellow", "lime", "red"]
    titles = [
        "Laplacian of Gaussian",
        "Difference of Gaussian",
        "Determinant of Hessian",
    ]
    sequence = zip(blobs_list, colors, titles)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image)
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()

    plt.tight_layout()
    plt.show()


# In[5]:


y = test_log(adjusted)
y

# In[6]:


z = test_log(original)


# In[55]:


def flame_blobs(image_path, gamma, plot_results=True):
    import time

    import numpy as np
    import cv2

    import matplotlib.pyplot as plt

    plt.rcdefaults()

    from math import sqrt
    from skimage.feature import blob_dog, blob_log, blob_doh
    from skimage.color import rgb2gray

    start = time.time()

    # Adjust the image
    image = cv2.imread(image_path, 1)
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    corrected = cv2.LUT(image, table)

    image_gray = rgb2gray(corrected)

    blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=0.1)

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=0.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=0.01)

    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ["yellow", "lime", "red"]
    titles = [
        "Laplacian of Gaussian",
        "Difference of Gaussian",
        "Determinant of Hessian",
    ]
    sequence = zip(blobs_list, colors, titles)

    res_counts = {}

    # only display the results if specified
    if not plot_results:
        for idx, (blobs, color, title) in enumerate(sequence):
            res_counts[title] = len(blobs)
        end = time.time()
        return res_counts, (end - start) * 1000

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image)
        res_counts[title] = len(blobs)
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()

    plt.tight_layout()
    plt.show()

    end = time.time()
    return res_counts, (end - start) * 1000


# In[56]:


abc = flame_blobs(image_path, 3, plot_results=False)

# In[58]:


abc

# In[121]:


for filename in os.listdir(imagepath):
    image_name = imagepath + filename
    res = flame_blobs(image_name, 3)
    print(res[0])

# In[ ]:


# In[22]:


for filename in os.listdir(imagepath):
    image_name = imagepath + filename
    res = flame_blobs(image_name, 3, plot_results=False)
    print(res[1])
    break

# In[ ]:


# # funcX

# In[24]:


from funcx.sdk.client import FuncXClient

# In[73]:


local_ep = "4325781a-fcfc-4dac-9017-aa5bf97db85b"
theta_ep = "f3d6b327-d262-43a2-96da-0dbf1f5468b2"
cooley_ep = "2bf4b19b-eaec-42b2-a191-b1542f3cc868"

fxc = FuncXClient()


# In[70]:


def hello_world(name):
    return f"Hello, {name}"


hello_func = fxc.register_function(hello_world, description="Test hello world.")
print(hello_func)

# In[71]:


name = "Ryan"
res = fxc.run(name=name, endpoint_id=local_ep, function_id=hello_func)

# In[72]:


fxc.get_result(res)

# In[74]:


name = "Cooley"
res = fxc.run(name=name, endpoint_id=cooley_ep, function_id=hello_func)

# In[75]:


fxc.get_result(res)

# In[63]:


flame_func = fxc.register_function(flame_blobs, description="Flame function.")
print(flame_func)

# Run locally

# In[64]:


input_loc = "/home/ryan/src/MERF/FlameSpray/RawData/R-233_5-8-6_000114.T000.D000.P000.H000.PLIF1.TIF"

# In[65]:


fxres = fxc.run(
    input_loc, gamma=3, plot_results=False, endpoint_id=local_ep, function_id=flame_func
)

# In[67]:


fx_result = fxc.get_result(fxres)

# In[68]:


fx_result

# Run on Cooley

# In[76]:


input_loc = "/home/rchard/DLHub/ryan/flame/RawData/R-233_5-8-6_000119.T000.D000.P000.H000.PLIF1.TIF"

# In[77]:


fxres = fxc.run(
    input_loc,
    gamma=3,
    plot_results=False,
    endpoint_id=cooley_ep,
    function_id=flame_func,
)

# In[80]:


fx_result = fxc.get_result(fxres)

# In[81]:


fx_result

# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# # PCA

# In[106]:


from sklearn.preprocessing import StandardScaler

# In[167]:


labels_fin = []
for i in X:
    if i[0] < 10 or i[0] > 450:
        labels_fin.append("Spill")
    else:
        labels_fin.append("No Spill")

# In[168]:


df = pd.DataFrame(columns=["B1", "B2", "B3", "Type"])
i = 0
for arr in X:
    df = df.append(
        {"B1": arr[0], "B2": arr[1], "B3": arr[2], "Type": labels_fin[i]},
        ignore_index=True,
    )
    i = i + 1

# In[169]:


df

# In[170]:


features = ["B1", "B2"]
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:, ["Type"]].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

# In[171]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(
    data=principalComponents, columns=["principal component 1", "principal component 2"]
)

# In[172]:


finalDf = pd.concat([principalDf, df[["Type"]]], axis=1)

# In[173]:


fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("Principal Component 1", fontsize=12)
ax.set_ylabel("Principal Component 2", fontsize=12)
ax.set_title("2 Component Principal Component Analysis (PCA)", fontsize=13)
targets = ["No Spill", "Spill"]
colors = ["r", "b"]
for target, color in zip(targets, colors):
    indicesToKeep = finalDf["Type"] == target
    ax.scatter(
        finalDf.loc[indicesToKeep, "principal component 1"],
        finalDf.loc[indicesToKeep, "principal component 2"],
        c=color,
        s=50,
    )
ax.legend(targets)
ax.grid()

# In[ ]:


import os

# In[36]:


training = []
labels = []

# In[12]:


for filename in os.listdir(imagepath):
    image = imagepath + filename
    image = cv2.imread(image)
    plt.imshow(image)
    plt.show()
for filename in os.listdir(imagepath):
    image = imagepath + filename
    image = cv2.imread(image)
    image = image[1125:1300, 1700:2000]
    plt.imshow(image)
    plt.show()

# In[37]:


for filename in os.listdir(imagepath):
    x = []
    image = imagepath + filename
    print(image)
    image = cv2.imread(image)
    s1 = LOG(image[250:600, 1000:2000])
    s2 = LOG(image[1000:1250, 2250:2500])
    s3 = LOG(image[1900:2000, 1000:2000])
    s4 = LOG(image[1125:1300, 1700:2000])
    avg = (s1 + s2 + s3 + s4) / 4
    training.append(avg)

# In[38]:


training

# In[49]:


xaxis = []
for item in training:
    xaxis.append(random.uniform(1, 2))

# In[51]:


# In[ ]:


# In[55]:


import numpy as np
import matplotlib.pyplot as plt

plt.scatter(xaxis, training, color="brown")
plt.xlabel("Length to Randomly Spread Images")
plt.ylabel("Average Grid Count")
# plt.title(label='Laplacian Blob Detection Count for SEM Images')
# brown_patch = mpatches.Patch(color='brown', label='Blurry Images')
# blue_patch =mpatches.Patch(color='lightblue', label='Focused Images')
# plt.axhline(y=350, color='green', linestyle='-', linewidth = 3.5)
# plt.axhline(y=500, color='grey', linestyle='-')
# plt.axhline(y=200, color='grey', linestyle='-')
# plt.legend(handles=[brown_patch, blue_patch])

plt.show()

# In[28]:


labels = []
for filename in os.listdir(imagepath):
    x = []
    image = imagepath + filename
    print(image)
    image = cv2.imread(image)
    num = LOG(image)
    if num < 10 or num > 450:
        labels.append("Spill!")
    else:
        labels.append("No Spill!")

# In[ ]:


# In[34]:


rf_df = pd.DataFrame(columns=["B1", "B2", "B3", "B4", "Type"])
i = 0
for arr in training:
    rf_df = rf_df.append(
        {"B1": arr[0], "B2": arr[1], "B3": arr[2], "B4": arr[3], "Type": labels[i]},
        ignore_index=True,
    )
    i = i + 1

# In[ ]:


# In[30]:


X = training
y = labels

# In[32]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# In[33]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# In[ ]:


# In[ ]:


# In[ ]:


# # K-Means Clustering

# In[ ]:


# In[102]:


# Getting the values and plotting it
f1 = df["B1"].values
f2 = df["B2"].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c="black", s=7)


# In[104]:


# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


# In[105]:


# Number of clusters
k = 2
# X coordinates of random centroids
C_x = np.random.randint(0, np.max(X) - 20, size=k)
# Y coordinates of random centroids
C_y = np.random.randint(0, np.max(X) - 20, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print(C)

# In[174]:


# Plotting along with the Centroids
plt.scatter(f1, f2, c="#050505", s=7)
plt.scatter(C_x, C_y, marker="*", s=200, c="g")

# In[ ]:


# In[ ]:


# In[ ]:
