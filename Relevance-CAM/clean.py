import os

save_dir_fig1 = '../data/R_CAM_results/figs/heatmaps'
save_dir_fig2 = '../data/R_CAM_results/figs/segmented_imgs'
save_dir_np1 = '../data/R_CAM_results/np/heatmaps'
save_dir_np2 = '../data/R_CAM_results/np/segmented_imgs'

dirs = [save_dir_fig1, save_dir_fig2, save_dir_np1, save_dir_np2]
methods = ['/Grad_CAM', '/Grad_CAM++', '/Score_CAM', '/Relevance_CAM']


for d in dirs:
    for m in methods:
        DIR = d+m
        print(f"cleaning directory {DIR}")
        for f in os.listdir(DIR):
            os.remove(os.path.join(DIR, f))