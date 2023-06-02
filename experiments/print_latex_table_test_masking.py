PATH = "/mnt/c/Users/Elia/Desktop/rp_item"
import numpy as np
import os
# get all files in PATH (all rp-item files of all masking types related experiments)
files = os.listdir(PATH)
masking_types = {}
for file in files:
    name = file.split("_")[-1]
    name = name[:-4]
    if name not in masking_types.keys():
        masking_types[name] = []
final_dict = {}
for masking_type in masking_types.keys():
    rp_item = np.load(PATH + "/rp-item_" + masking_type + ".npy", allow_pickle=True)
    corruptions = {}
    for item in rp_item:
        name = item["name"]
        name = name.split("/")[-1]
        if name.split("_")[-1][:8] == "original":
            name = "no corruption"
        else:
            name = name.split("_")[-2] + " " + name.split("_")[-1]
            name = name[:-4]
        if name not in corruptions.keys():
            corruptions[name] = {}
            a = corruptions[name]
            a["chamfer"] = []
            a["p2s"] = []
        a = corruptions[name]
        a["chamfer"].append(item["vals"][0])
        a["p2s"].append(item["vals"][1])

    chamfer_base = np.mean(corruptions["no corruption"]["chamfer"])*10
    p2s_base = np.mean(corruptions["no corruption"]["p2s"])*10
    for key in corruptions.keys():
        a = corruptions[key]
        chamfer = a["chamfer"]
        p2s = a["p2s"]
        chamfer_mean = np.mean(chamfer)*10
        p2s_mean = np.mean(p2s)*10
        chamfer_diff = chamfer_mean - chamfer_base
        p2s_diff = p2s_mean - p2s_base
        final_dict[key + masking_type] = [chamfer_mean, p2s_mean, chamfer_diff, p2s_diff]
    masking_types[masking_type] = corruptions
print(" & ", end="") 
for masking_type in masking_types.keys():
    print(masking_type + " & ", end="")
print("\\\\")

for key in corruptions.keys():
    print(key + " & ", end="")
    for masking_type in masking_types.keys():
        a = final_dict[key + masking_type]
        chamfer_diff = a[0]
        p2s_diff = a[1]
        print(str(round(chamfer_diff, 4)) + " & " + str(round(p2s_diff, 4)) + " & ", end="")
    print("\\\\")



    



