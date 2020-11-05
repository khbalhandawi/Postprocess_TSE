import numpy as np
import os

#==============================================================================
# MAIN CALL
if __name__ == "__main__":
    
    plot_index = 0
    n = np.arange(100,1000,100)

    for n_points in n:
        plot_index += 1
        command = "python post_process_DOE.py %i %i" %(n_points,plot_index)
        print(command)
        os.system(command)

    n = np.arange(1000,10000,1000)

    for n_points in n:
        plot_index += 1
        command = "python post_process_DOE.py %i %i" %(n_points,plot_index)
        print(command)
        os.system(command)

    n = np.arange(10000,100000,10000)

    for n_points in n:
        plot_index += 1
        command = "python post_process_DOE.py %i %i" %(n_points,plot_index)
        print(command)
        os.system(command)