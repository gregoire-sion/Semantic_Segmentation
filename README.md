  File "/home/gsionsua/Work_bis/KalmanNet_new/KalmanNet_TSP-main(1)/KalmanNet_TSP-main/main_linear_CA.py", line 182, in <module>
    Plot.plotTraj_CA(test_target, KF_out, KNet_out, dim=0, file_name=PlotfolderName+PlotfileName0)#Position
  File "/home/gsionsua/Work_bis/KalmanNet_new/KalmanNet_TSP-main(1)/KalmanNet_TSP-main/Plot.py", line 348, in plotTraj_CA
    plt.plot(x_plt, rtsnet_out[0][0,:].detach().numpy(), label=legend[0])
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
