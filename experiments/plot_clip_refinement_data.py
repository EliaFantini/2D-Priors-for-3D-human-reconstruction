import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

PICKLE_FILE_PATH_SEQUENTIAL = "/mnt/c/Users/Elia/Desktop/tta_clip/results/day_2023_05_31_time_19_28_31/corruption_types.pickle"
PICKLE_FILE_PATH_AVERAGE = "/mnt/c/Users/Elia/Desktop/corruption_types.pickle"



def save_big_plot(sequential = True):

    SAVE_FOLDER = "/mnt/c/Users/Elia/Desktop/rp_item"
    # open the pickle file
    # open the pickle file
    if sequential:
        with open(PICKLE_FILE_PATH_SEQUENTIAL, 'rb') as f:
            corruption_types = pickle.load(f)
    else:
        with open(PICKLE_FILE_PATH_AVERAGE, 'rb') as f:
            corruption_types = pickle.load(f)
    corruptions = corruption_types.keys()

    fig, axs = plt.subplots(4, len(corruptions), figsize=(40, 15), sharey='row')



    

    objs = corruption_types["zoom_blur"].keys()
    for i, corruption in enumerate(corruptions):
        clip_loss_lists = []
        psnr_lists = []
        ssim_lists = []
        clip_sim_lists = []
        for obj in objs:
            a = corruption_types[corruption]
            listt= a[obj]
            clip_loss_lists.append(listt[0])
            psnr_lists.append(listt[1])
            ssim_lists.append(listt[2])
            clip_sim_lists.append(listt[3])
        clip_loss_lists = np.array(clip_loss_lists)
        psnr_lists = np.array(psnr_lists)
        ssim_lists = np.array(ssim_lists)
        clip_sim_lists = np.array(clip_sim_lists)
        clip_loss_mean = np.mean(clip_loss_lists, axis=0)
        clip_loss_std = np.std(clip_loss_lists, axis=0)
        psnr_mean = np.mean(psnr_lists, axis=0)
        psnr_std = np.std(psnr_lists, axis=0)
        ssim_mean = np.mean(ssim_lists, axis=0)
        ssim_std = np.std(ssim_lists, axis=0)
        clip_sim_mean = np.mean(clip_sim_lists, axis=0)
        clip_sim_std = np.std(clip_sim_lists, axis=0)

       


       
        window_size = 4  

        clip_loss_mean = savgol_filter(clip_loss_mean, window_size, 3)  
        clip_loss_std = savgol_filter(clip_loss_std, window_size, 3)  
        psnr_mean = savgol_filter(psnr_mean, window_size, 3)  
        psnr_std = savgol_filter(psnr_std, window_size, 3)  
        ssim_mean = savgol_filter(ssim_mean, window_size, 3)  
        ssim_std = savgol_filter(ssim_std, window_size, 3)  
        clip_sim_mean = savgol_filter(clip_sim_mean, window_size, 3)  
        clip_sim_std = savgol_filter(clip_sim_std, window_size, 3)  




        colors = ["red", "blue", "green", "yellow"]
        # use a different color for each plot
        
        #each row should share the y axis
        name = corruption.split("_")[0] + " " + corruption.split("_")[1]
        if name == "roll 80":
            name = "camera roll"
        if name == "render input":
            name = "no corruption"

        
        axs[0, i].plot(clip_loss_mean, label="clip loss", color=colors[0])
        # change the color of this plot
        axs[0, i].set_title(name)
        axs[0, i].set_ylabel("clip loss")
        axs[0, i].legend()
        axs[1, i].plot(psnr_mean, label="psnr", color=colors[1])
        axs[1, i].set_ylabel("psnr")
        axs[1, i].legend()
        axs[2, i].plot(ssim_mean, label="ssim", color=colors[2])
        axs[2, i].set_ylabel("ssim")
        axs[2, i].legend()
        axs[3, i].plot(clip_sim_mean, label="clip gt diff", color=colors[3])
        axs[3, i].set_ylabel("clip gt diff")
        axs[3, i].set_xlabel("epochs")
        axs[3, i].legend()
    plt.savefig(SAVE_FOLDER + "/plot.png")
    plt.close()

def save_smaller_plot_of_average_sequeantial_comparison():

    PATHS = [PICKLE_FILE_PATH_SEQUENTIAL, PICKLE_FILE_PATH_AVERAGE]

    SAVE_FOLDER = "/mnt/c/Users/Elia/Desktop/rp_item"


    fig, axs = plt.subplots(4, 2, figsize=(10, 15), sharey='row')
    
    for i,path in enumerate(PATHS):
        with open(path, 'rb') as f:
                corruption_types = pickle.load(f)
        # get only the first corruption type
        corruptions = []
        corruptions.append(list( corruption_types.keys()))



        

        objs = corruption_types["zoom_blur"].keys()
        for corruption in corruptions:
            clip_loss_lists = []
            psnr_lists = []
            ssim_lists = []
            clip_sim_lists = []
            for obj in objs:
                a = corruption_types["bit_error"]
                listt= a[obj]
                clip_loss_lists.append(listt[0])
                psnr_lists.append(listt[1])
                ssim_lists.append(listt[2])
                clip_sim_lists.append(listt[3])
            clip_loss_lists = np.array(clip_loss_lists)
            psnr_lists = np.array(psnr_lists)
            ssim_lists = np.array(ssim_lists)
            clip_sim_lists = np.array(clip_sim_lists)
            clip_loss_mean = np.mean(clip_loss_lists, axis=0)
            clip_loss_std = np.std(clip_loss_lists, axis=0)
            psnr_mean = np.mean(psnr_lists, axis=0)
            psnr_std = np.std(psnr_lists, axis=0)
            ssim_mean = np.mean(ssim_lists, axis=0)
            ssim_std = np.std(ssim_lists, axis=0)
            clip_sim_mean = np.mean(clip_sim_lists, axis=0)
            clip_sim_std = np.std(clip_sim_lists, axis=0)




           
            window_sizes = [15,4]
            window_size = window_sizes[i] 

            clip_loss_mean = savgol_filter(clip_loss_mean, window_size, 3) 
            clip_loss_std = savgol_filter(clip_loss_std, window_size, 3) 
            psnr_mean = savgol_filter(psnr_mean, window_size, 3) 
            psnr_std = savgol_filter(psnr_std, window_size, 3) 
            ssim_mean = savgol_filter(ssim_mean, window_size, 3) 
            ssim_std = savgol_filter(ssim_std, window_size, 3) 
            clip_sim_mean = savgol_filter(clip_sim_mean, window_size, 3) 
            clip_sim_std = savgol_filter(clip_sim_std, window_size, 3) 





            colors = ["red", "blue", "green", "yellow"]

            names = ["Sequential", "Average"]

            
            axs[0, i].plot(clip_loss_mean, label="clip loss", color=colors[0])

            axs[0, i].set_title(names[i])
            axs[0, i].set_ylabel("clip loss")
            axs[0, i].legend()
            axs[1, i].plot(psnr_mean, label="psnr", color=colors[1])
            axs[1, i].set_ylabel("psnr")
            axs[1, i].legend()
            axs[2, i].plot(ssim_mean, label="ssim", color=colors[2])
            axs[2, i].set_ylabel("ssim")
            axs[2, i].legend()
            axs[3, i].plot(clip_sim_mean, label="clip gt diff", color=colors[3])
            axs[3, i].set_ylabel("clip gt diff")
            axs[3, i].set_xlabel("epochs")
            axs[3, i].legend()
    plt.savefig(SAVE_FOLDER + "/plot.png")
    plt.close()







if __name__ == '__main__':
    #save_smaller_plot_of_average_sequeantial_comparison()
    save_big_plot()