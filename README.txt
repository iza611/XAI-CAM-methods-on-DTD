This code consists of three stages:

1. Transfer learning to adapt a pretrained ResNet50 model to the Describable Textures Dataset (DTD).

2. Applying Grad-CAM, Grad-CAM++, Score-CAM, and Relevance-CAM to trained model with 10 example images from the DTD.

3. Evaluating the performance of the methods using the Quantus library. This includes:
- Comparing the results from stage 2 side by side
- Calculating five chosen metrics
- Comparing the results of the metrics.

########################################################################################################################################

NOTE: Running all of the code may take a long time. On my laptop (Intel(R) Core(TM) i5-4300U CPU @ 1.90GHz 2.50 GHz, 8GB RAM) 
it took me few days. If you do not have much time available, the results of each stage are saved as follows:

-Results from stage 1 (transfer learning) can be found in the /train_resnet_with_transfer_learning/transfer_learning.ipynb file.
-Results from stage 2 (application of Grad-CAM, Grad-CAM++, Score-CAM, and Relevance-CAM) are saved in the 
/Relevance-CAM/data/R_CAM_results_layer2 and /Relevance-CAM/data/R_CAM_results_layer4 directories.
-Results from stage 3 (evaluation using the Quantus library) can be found in the /Relevance-CAM/evaluate1.ipynb, 
/Relevance-CAM/data/results.json (NOT in /Relevance-CAM/evaluate2.ipynb), and /Relevance-CAM/evaluate3.ipynb files.
	
########################################################################################################################################

To run the code, please follow these steps:

1. Make sure that Anaconda, Anaconda Prompt, and VS Code are installed on your computer. If they are not installed, please follow the 
instructions at:

	https://docs.anaconda.com/anaconda/install/index.html
	https://code.visualstudio.com/download

2. Open the Anaconda Prompt, navigate to the directory where this README file and all of the code is located, and run the following commands:

	conda create --name <environment_name> --file req.txt
	conda activate <environment_name>

Replace <environment_name> with a name of your choice. This will create a new environment with the same packages installed as the ones I used. 
To make sure everything is working correctly, run conda list and compare the packages and Python version to those listed in the req.txt file.

3. Download the DTD dataset from https://www.robots.ox.ac.uk/~vgg/data/dtd/. Unzip it and place it inside the /data folder:

	/data/dtd

4. Open the entire 'code' folder in VS Code. Make sure that the kernel is set to the conda environment you created in step 2.

5. To perform stage 1, run the code in the /train_resnet_with_transfer_learning/transfer_learning.ipynb file.

6. To perform stage 2, run the code in the /Relevance-CAM/run.ipynb file.

7. To perform the evaluation, run the code in the /Relevance-CAM/evaluate1.ipynb, /Relevance-CAM/evaluate2.ipynb, and /Relevance-CAM/evaluate3.ipynb files.

########################################################################################################################################

Code references:
[4]	Hedström, A. et al. (2022) “Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural Network Explanations,” arXiv [Preprint]. Available at: https://doi.org/https://doi.org/10.48550/arXiv.2202.06861.
[5]	Lee, J.R. et al. (2021) “Relevance-cam: Your model already knows where to look,” 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) [Preprint]. Available at: https://doi.org/10.1109/cvpr46437.2021.01470.