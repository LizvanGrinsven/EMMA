#%%

import numpy as np
import pandas as pd
from scipy.optimize import nnls
import matplotlib.pyplot as plt
import xlsxwriter
import pathlib
path = pathlib.Path(__file__).parent.resolve()
path_plot = path / 'pimpmyplot_large.mplstyle'
plt.style.use(path_plot)

#%% FILL OUT

m = 3 # number of end-members 
B_lab = 40 # lab induced field strength 
q = 5000 # number of iterations 

# Do you want to remove the outliers?
remove_outliers = 'Yes' # Fill in Yes or No
outliers_min = 0 # If yes: What is the lower limit (check histogram)
outliers_max = 9  # If yes: What is the upper limit (check histogram)

# Do you want to remove the overprint?
remove_overprint = 'No' # Fill in Yes or No
overprint_removed = 2.5 # If yes: Give NRM demagnetization step from which overprint is removed

#Do you randomly want to exclude a percentage of the data in the unmixing phase and calculate the paleointensity for this excluded group
test_random = 'No' # Fill in Yes or No
percentage_test_random = 30 # Fill in the percentage of data to remove, max 100

# choose a value for alpha_1 and alpha_2 (must sum to 1), default alpha_1 = alpha_2 = 0.5
alpha_1 = 0.5 # weight for A towards NRM unmixing
alpha_2 = 0.5 # weight for A towards ARM unmixing 

# Load in excel data file
path_data = path / 'input_synthetic.xlsx'
file_loc = path_data
which_data_set = 'Data' # fill in Numerical (only works with 3 end-members) or Data
percentage_noise_stdv = 0 # fill in 0 for no noise

# eps represents some small number , to avoid division by zero
eps = 10**-9
#%% creating numerical data-set


if which_data_set == 'Numerical':
    # Numerical model input
    n = 255
    S_x_input = np.matrix([[0.0254128202346398, 0.0192149729966965, 0.0125697248618582, 0.00712823926248334, 0.00465129504633213, 0.00274148694065965, 0.00169855123359381, 0.000944645733509271, 0.000440643634303486, 9.54391421641192e-05, 1.88626811158089e-05, 6.04560974384903e-06, 2.87896733627421e-06, 1.88156160619516e-06, 1.60715203717647e-06, 6.52348807897405e-07, 6.52348807897405e-07, 0], 
                  [0.0571785968159724, 0.0571785968159724, 0.0571785968159724, 0.0571785968159724, 0.0511383991883729, 0.0364276824426319, 0.0225142420085124, 0.0121718335535526, 0.00554406064305267, 0.00127646053496491, 0.000277971364906962, 9.48341629345917e-05, 4.50407237896227e-05, 2.80692503797647e-05, 1.92090006820587e-05, 5.73015270408536e-06, 5.73015270408536e-06, 0], 
                  [0.0793378149790575, 0.0793378149790575, 0.0793378149790575, 0.0783860461334386, 0.0777888887197825, 0.0777888887197825, 0.0777888887197825, 0.0760763186701476, 0.0716924272539771, 0.0614723215606749, 0.0495505072123399, 0.0400208874741691, 0.0321996399802047, 0.0267505253489354, 0.0179058524256274, 0.00950827142071631, 0.00390201488611561, 0]])
    S_y_input = np.matrix([[0, 0.00529097997090905, 0.00966131673863859, 0.0136098201878304, 0.0171443801027178, 0.0207715119040851, 0.0224016072215791, 0.0229561482344192, 0.0229561482344192, 0.0229561482344192, 0.0229561482344192, 0.0229561482344192, 0.0229607194496053, 0.0229650746583995, 0.0231867723681946, 0.0235878689059207, 0.0249420811389439, 0.0250000000000000], 
                  [0, 5.69939071353337e-05, 0.000132205324869256, 0.000353157359651385, 0.000937161521100360, 0.00297970209519448, 0.00573896750635424, 0.00873672739347442, 0.0119521048817004, 0.0160685468934525, 0.0194519056961943, 0.0217564405101545, 0.0233528536040274, 0.0242923737697005, 0.0250000000000000, 0.0250000000000000, 0.0250000000000000, 0.0250000000000000], 
                  [0, 1.05834169697356e-05, 3.41650053753493e-05, 0.000164328475412998, 0.000609511149944067, 0.00188650202255817, 0.00306054796377527, 0.00409823033183112, 0.00498918451007610, 0.00626440178595607, 0.00774725809229816, 0.00915266073825647, 0.0106249944516446, 0.0119191758854556, 0.0148868710141033, 0.0204315053146024, 0.0235210059312522, 0.0250000000000000]])
    A_input = np.random.rand(n,m)
    A_input = A_input / np.sum(A_input, 1).reshape(n,1)
    X = np.matmul(A_input, S_x_input)
    
    # adding Gaussian noise
    noise_X = np.random.normal(0, (percentage_noise_stdv/100)*X.mean(), X.shape)
    X = X + noise_X
    Y = np.matmul(A_input, S_y_input)
    noise_Y = np.random.normal(0, (percentage_noise_stdv/100)*Y.mean(), Y.shape)
    Y = Y + noise_Y
                
    B_x_ref = np.random.uniform(low=0, high=100, size=(n,1))
    B_x_ref.astype(float)
    x_as_NRM = np.array([0, 2.5, 5, 7.5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 150, 225, 300]).flatten()
    x_as_ARM = np.array([0, 2.5, 5, 7.5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 150, 225, 300]).flatten()
    min_field = np.full((n,1),1)
    locations = np.full((n,1), 'Numerical')
    k_x = X.shape[1]
    k_y = Y.shape[1]
    steps = np.arange(n).reshape(n,1)
    
    # apply boundary conditions
    for j in range (n):
        X[j, -1] = 0
        Y[j, 0] = 0
        Y[j, -1] = 0.025
        
        for o in range (k_x - 1): 
            if X[j, k_x-o-2] < X[j, k_x-o-1]: 
                X[j, k_x-o-2] = X[j, k_x-o-1] 
                 
        for o in range (k_y - 1): 
            if Y[j, -2-o] > Y[j, -1-o]: 
                Y[j, -2-o] =Y[j, -1-o] 
    
    X_unmixer = X
    Y_unmixer = Y
    B_x_ref_unmixer = B_x_ref
    n_unmixer = n
    
    X_dataset = X
    Y_dataset = Y
    B_x_ref_dataset = B_x_ref
    n_dataset = n
    
    #Plotting Numerical model input 
    fig, axd = plt.subplot_mosaic([['a', 'c'], ['b', 'd']], constrained_layout = True, figsize=(7.2,4.8), dpi=1200)
    fig.suptitle('Numerical model input')
    for i in range (m):    
        axd['a'].plot(x_as_NRM, np.array(S_x_input[i,:]).flatten())
        axd['b'].plot(x_as_ARM, np.array(S_y_input[i,:]).flatten()) 
    for i in range (n):
        axd['c'].plot(x_as_NRM, np.array(X[i,:]).reshape(k_x,1) * B_x_ref[i])
        axd['d'].plot(x_as_ARM, np.array(Y[i,:]).reshape(k_y,1) * B_lab)
            
    axd['a'].annotate('a.', (0.92, 0.92), xycoords='axes fraction' , fontsize=10)
    axd['b'].annotate('b.', (0.92, 0.03), xycoords='axes fraction' , fontsize=10)
    axd['c'].annotate('c.', (0.92, 0.92), xycoords='axes fraction' , fontsize=10)
    axd['d'].annotate('d.', (0.92, 0.03), xycoords='axes fraction' , fontsize=10)
    
    axd['a'].set_title(f'{m} end-members INPUT: NRM demagnetization')
    axd['b'].set_title(f'{m} end-members INPUT: ARM acquisition')
    axd['c'].set_title('NRM demagnetization data')
    axd['d'].set_title('ARM acquisition data')
    
    axd['a'].set_xlabel('Demagnetization steps [mT]')
    axd['a'].set_ylabel('Magnetization[-]')
    axd['b'].set_xlabel('Remagnetization steps [mT]')
    axd['b'].set_ylabel('Magnetization [-]')
    axd['c'].set_xlabel('Demagnetization steps [mT]')
    axd['c'].set_ylabel('Magnetization [-]')
    axd['d'].set_xlabel('Remagnetization steps [mT]')
    axd['d'].set_ylabel('Magnetization [-]')

    plt.show()

#%% pre process the data


if which_data_set == "Data":
    # load in the data
    # NRM demagnetization data-set 
    X = pd.read_excel(file_loc, index_col=None, na_values=['NA'], usecols="H:Y", skiprows=0) 
    X = X.to_numpy()
    # ARM acquisition data-set 
    Y = pd.read_excel(file_loc, index_col=None, na_values=['NA'], usecols="AA:AR", skiprows=0) 
    Y = Y.to_numpy()
    # the known paleointensities for each sample 
    B_x_ref = pd.read_excel(file_loc, index_col=None, na_values=['NA'], usecols="F", skiprows=0) 
    B_x_ref = B_x_ref.to_numpy() 
    # NRM demagnetization steps 
    x_as_NRM = pd.read_excel(file_loc, index_col=None, na_values=['NA'], header=None, usecols="H:Y", nrows=1) 
    x_as_NRM = x_as_NRM.to_numpy().flatten()
    # ARM acquisition steps 
    x_as_ARM = pd.read_excel(file_loc, index_col=None, na_values=['NA'], header=None, usecols="AA:AR", nrows=1) 
    x_as_ARM = x_as_ARM.to_numpy().flatten()
    # import min field step, for removing overprint
    min_field = pd.read_excel(file_loc, index_col=None, na_values=['NA'], usecols="D", skiprows=0)
    min_field = min_field.to_numpy()
    # import field locations
    locations = pd.read_excel(file_loc, index_col=None, na_values=['NA'], usecols="B", skiprows=0)
      
    k_x = X.shape[1] # number of measurement steps for NRM demagneitzion data-set 
    k_y = Y.shape[1] # number of measurement steps for ARm acquisition data-set 
    n = X.shape[0] # number of samples 
    
    # make NRM demagnetizaiton and ARM acquisition data end and begin at zero
    steps_remove_zero = []
    for i in range (n): 
        X[i, :] -= X[i, k_x-1] 
        Y[i, :] -= Y[i, 0] 
    
    # make NRM demagnetization and ARM acquisition data monotonic     
        for j in range(k_x - 1):
            if X[i, k_x-2-j] < X[i, k_x-1-j]:
                X[i, k_x-2] = X[i, k_x-1]
            
            if Y[i, j+1] < Y[i, j]:
                Y[i, j+1] = Y[i, j]
    
    # remove the X and Y data that is all zero
        if  (X[i,0] == 0) or (Y[i,(k_y-1)] == 0 ):
            steps_remove_zero.append(i)
    
    # normalize NRM demagnetization and ARM acquisition data by maximum value of the ARM acquisition measurement            
        X[i, :] = X[i, :] / (Y[i, k_y-1]+eps)
        Y[i, :] = Y[i, :] / (Y[i, k_y-1]+eps)      
    
    # remove all infs and nans from data
    B_x_ref = B_x_ref[~np.isnan(X).any(axis=1)]
    Y = Y[~np.isnan(X).any(axis=1)]   
    X = X[~np.isnan(X).any(axis=1)]
    B_x_ref = B_x_ref[~np.isnan(Y).any(axis=1)]
    Y = Y[~np.isnan(Y).any(axis=1)]   
    X = X[~np.isnan(Y).any(axis=1)]
    B_x_ref = B_x_ref[~np.isinf(X).any(axis=1)]
    Y = Y[~np.isinf(X).any(axis=1)]   
    X = X[~np.isinf(X).any(axis=1)]
    B_x_ref = B_x_ref[~np.isinf(Y).any(axis=1)]
    Y = Y[~np.isinf(Y).any(axis=1)]   
    X = X[~np.isinf(Y).any(axis=1)]
    n = X.shape[0]
    
    #remove outliers
    Xsteps_minus_outliers = []
    if remove_outliers == 'No' :
        Xsteps_minus_outliers = np.arange(0,n,1)
    elif remove_outliers == 'Yes' :
        for i in range (n):
            if (X[i,1] > outliers_min) & (X[i,1] < outliers_max):
                Xsteps_minus_outliers.append(i)
                
    # divide by field strength
    for i in range (n): 
        X[i, :] = X[i, :] / B_x_ref[i]
        Y[i, :] = Y[i, :] / B_lab
    
    x_as_NRM_original = x_as_NRM.copy()
    # remove overprint    
    if remove_overprint == "No" :
        Xsteps_minus_overprint = np.arange(0,n,1)
        overprint_step = ([1])
    elif remove_overprint == 'Yes' :
        overprint_step = np.where(x_as_NRM == overprint_removed)[0] + 1
        Xsteps_minus_overprint = np.where((min_field>0) & (min_field<=overprint_step))[0]
        x_as_NRM = x_as_NRM[(overprint_step[0]-1):]    
        k_x = k_x - (overprint_step[0]-1)

    #remove the zero values first to plot the histogram of normalized NRMmax, before removing outliers etc.
    steps = []
    for i in range (n):
        if (i not in steps_remove_zero):
            steps.append(i)
            
    # define X, Y, B_x_ref and n with possible outliers and overprint removed 
    steps = []
    for i in range (n):
        if (i not in steps_remove_zero):
            steps.append(i)
    
    X_original = X[steps]
    Y_original = Y[steps]
    B_x_ref_original = B_x_ref[steps]
    n_original = X_original.shape[0]
    
    steps = []
    for i in range (n):
        if (i in Xsteps_minus_outliers) & (i in Xsteps_minus_overprint) & (i not in steps_remove_zero):
            steps.append(i)
    

    X_unmixer = X[steps]
    X_unmixer = X_unmixer[:,(overprint_step[0]-1):]
    X_dataset = X_unmixer
    Y_unmixer = Y[steps]
    Y_dataset = Y_unmixer
    B_x_ref_unmixer = B_x_ref[steps]
    B_x_ref_dataset = B_x_ref_unmixer
    n_unmixer = X_unmixer.shape[0]
    n_dataset = n_unmixer 
    
    # remove a percentage from the dataset at random in the unmixing stage step 1, and calculate the paleointensity for this removed part in step 2
    Xsteps_test_random = []
    Xsteps_test_random_unmixer = []
    steps_test_locations_half_unmixer = []
    steps_test_locations_half = []
    if test_random == 'Yes' :
        steps_test_random = np.random.randint(0,n,np.int((n/100) * percentage_test_random))
            
        for i in range (n):
            if (i in Xsteps_minus_outliers) & (i in Xsteps_minus_overprint) & (i not in steps_remove_zero) & (i not in steps_test_random):
                Xsteps_test_random_unmixer.append(i)
            if (i in Xsteps_minus_outliers) & (i in Xsteps_minus_overprint) & (i not in steps_remove_zero) & (i in steps_test_random):
                Xsteps_test_random.append(i)
                
        X_dataset = X[Xsteps_test_random]
        X_dataset = X_dataset[:,(overprint_step[0]-1):]
        Y_dataset = Y[Xsteps_test_random]
        B_x_ref_dataset = B_x_ref[Xsteps_test_random]
        n_dataset = X_dataset.shape[0]
            
        X_unmixer = X[Xsteps_test_random_unmixer]
        X_unmixer = X_unmixer[:,(overprint_step[0]-1):]
        Y_unmixer = Y[Xsteps_test_random_unmixer]
        B_x_ref_unmixer = B_x_ref[Xsteps_test_random_unmixer]
        n_unmixer = X_unmixer.shape[0]
        
        steps = Xsteps_test_random       
                    
    
    # Plot the histogram showing outliers, and the data before and after removing outliers
    fig, axd = plt.subplot_mosaic([['a.','a.', 'b.', 'c.'], ['a.', 'a.', 'd.', 'e.']], constrained_layout = True, figsize=(7.2,3), dpi=1200)
    #fig.suptitle('Removing outliers')
    
    axd['a.'].hist(X_original[:,0], bins=1000)
    axd['a.'].set_xlabel('Max NRM normalized by max ARM [-]')
    axd['a.'].set_ylabel('Number of samples')
    
    for i in range (n_original):
        axd['b.'].plot(x_as_NRM_original, X_original[i,:]*B_x_ref_original[i], linewidth=1)
        axd['c.'].plot(x_as_ARM, Y_original[i,:]*B_lab, linewidth=1)
        
    for i in range (n_unmixer):
        axd['d.'].plot(x_as_NRM, X_unmixer[i,:]*B_x_ref_unmixer[i], linewidth=1)
        axd['e.'].plot(x_as_ARM, Y_unmixer[i,:]*B_lab, linewidth=1)
        
    axd['a.'].annotate('a.', (0.95, 0.95), xycoords='axes fraction' , fontsize=10)
    axd['b.'].annotate('b.', (0.9, 0.85), xycoords='axes fraction' , fontsize=10)
    axd['c.'].annotate('c.', (0.9, 0.05), xycoords='axes fraction' , fontsize=10)
    axd['d.'].annotate('d.', (0.9, 0.85), xycoords='axes fraction' , fontsize=10)
    axd['e.'].annotate('e.', (0.9, 0.05), xycoords='axes fraction' , fontsize=10)
    axd['d.'].set_ylabel('Magnetization [-]')
    axd['d.'].set_xlabel('Demagnetization steps [mT]')
    axd['e.'].set_xlabel('Remagnetization steps [mT]')
    
    plt.show()
    
    
print('Number of samples:', n_unmixer)

#%% unmixing, find the optimal iteration/ local minimum

X = X_unmixer
Y = Y_unmixer
B_x_ref = B_x_ref_unmixer
n = n_unmixer

# make x and y matrix not devided by field strength
B_x_ref_matrix = np.zeros((n,n))
np.fill_diagonal(B_x_ref_matrix, B_x_ref)
x = np.matmul(B_x_ref_matrix, X)
x = np.matrix.transpose(x).reshape(k_x,n)
y = B_lab * Y
y = np.matrix.transpose(y).reshape(k_y,n)

# initial guess
S_x_old = np.zeros((m, k_x))
S_y_old = np.zeros((m, k_y))
for i in range (m):    
    S_x_old[i, :] = np.linspace((0.05 + (i * 0.001)), 0, k_x)
    S_y_old[i, :] = np.linspace(0, (0.025 + (i * 0.001)) , k_y)
    
A_old = np.ones((n, m))
A_old = A_old / np.sum(A_old, 1).reshape(n,1)

# plot initial guess
fig, axd = plt.subplot_mosaic([['a', 'b']], constrained_layout = True, figsize=(7.2,3), dpi=1200)
   
for i in range (m): 
    axd['a'].plot(x_as_NRM, np.array(S_x_old[i,:]).flatten())
    axd['b'].plot(x_as_ARM, np.array(S_y_old[i,:]).flatten())

axd['a'].set_title(f'Initial guess {m} end-members: NRM demagnetization')
axd['b'].set_title(f'Initial guess {m} end-members: ARM acquisition')
axd['a'].set_xlabel('Demagnetization steps [mT]')
axd['a'].set_ylabel('Normalized magnetization')
axd['b'].set_xlabel('Remagnetization steps [mT]')
axd['b'].set_ylabel('Normalized magnetization')
plt.show()

# defining matrices
iterations = np.zeros(q).reshape(q,1)
B_x_diff_mean_it = np.zeros(q).reshape(q,1)
B_x_diff_abs_mean_it = np.zeros(q).reshape(q,1)
X_diff_mean_percentage_it = np.zeros((q,1))
Y_diff_mean_percentage_it = np.zeros((q,1))
X_euclean_it = np.zeros(q)
Y_euclean_it = np.zeros(q)

# start iteration
for i in range (q):
    
    iterations[i]=i
    
    # calculate new A for X and Y dataset
    A = alpha_1 * np.multiply(A_old, (np.matmul(Y, np.transpose(S_y_old))
        / (np.matmul(np.matmul(A_old, S_y_old), np.transpose(S_y_old)) + eps))) \
    + alpha_2 * np.multiply(A_old, (np.matmul(X, np.transpose(S_x_old))
        / (np.matmul(np.matmul(A_old, S_x_old), np.transpose(S_x_old)) + eps)))
    
    # calculate new E
    S_x = np.multiply(S_x_old, (np.matmul(np.transpose(A_old), X) / (np.matmul(np.matmul(np.transpose(A_old), A_old), S_x_old) + eps)))
    S_y = np.multiply(S_y_old, (np.matmul(np.transpose(A_old), Y) / (np.matmul(np.matmul(np.transpose(A_old), A_old), S_y_old) + eps)))
    
    # calculate the error percentage 
    X_check = np.matmul(A, S_x)
    Y_check = np.matmul(A, S_y)
    X_diff_mean_percentage_it[i] = np.mean(np.absolute(X_check - X)) / np.mean(X) * 100
    Y_diff_mean_percentage_it[i] = np.mean(np.absolute(Y_check - Y)) / np.mean(Y) * 100
    
    # calculate the squared Euclean distance
    X_euclean = 0
    Y_euclean = 0
    for j in range (n):
        for o in range (k_x):
            X_euclean = X_euclean + (X[j,o] - X_check[j,o])**2
        for o in range (k_y):
            Y_euclean = Y_euclean + (Y[j,o] - Y_check[j,o])**2
    
    X_euclean_it[i] = X_euclean / (n * k_x)
    Y_euclean_it[i] = Y_euclean / (n * k_y)
    
    # the new end-members become the old end-members
    S_x_old = S_x
    S_y_old = S_y
    A_old = A / np.sum(A, 1).reshape(n,1)

    # boundary conditions
    for j in range (m):
        S_x_old[j, -1] = 0
        S_y_old[j, 0] = 0
        S_y_old[j, -1] = 0.025
        
        for o in range (k_x - 1): 
            if S_x_old[j, k_x-o-2] < S_x_old[j, k_x-o-1]: 
                S_x_old[j, k_x-o-2] = S_x_old[j, k_x-o-1] 
                 
        for o in range (k_y - 1): 
            if S_y_old[j, -2-o] > S_y_old[j, -1-o]: 
                S_y_old[j, -2-o] = S_y_old[j, -1-o] 
                
    # calculation of paleointensity with the calculated end-members

    # transpose the end-members for the calculation of paleointensity
    s_x = np.transpose(S_x)
    s_y = np.transpose(S_y)
    s_y = s_y * B_lab
    
    # allocate space for matrix a and B_x
    a = np.zeros((m, n))
    B_x = np.zeros(n).reshape(n,1)
    
    #calculate paleointensity, done seperately for every measurement
    for j in range (n):
        # calculate B_x from A, E_x and X
        B_x[j] = np.linalg.lstsq( nnls(s_y, y[:, j].flat)[0].reshape(m,1), nnls(s_x, x[:, j].flat)[0].reshape(m,1),rcond=None)[0]
        
    # compare the results to the known paleointensity of every measurement
    B_x = B_x.reshape(n,1)
    B_x_diff = B_x - B_x_ref
    B_x_diff_mean_it[i] = np.mean(B_x_diff)
    B_x_diff_pos = np.absolute(B_x_diff)
    B_x_diff_abs_mean_it[i] = np.mean(B_x_diff_pos)

# optimal iteration chosen by best calculation of paleointensity    
B_x_min = min(B_x_diff_abs_mean_it)
B_x_min_iteration = np.where(B_x_diff_abs_mean_it==B_x_min)[0]

print('Optimal iteration:', B_x_min_iteration)

# plot the optimalization 
fig, axd = plt.subplot_mosaic([['a', 'b']], constrained_layout = True, figsize=(7.2,3), dpi=1200)
#fig.suptitle('Performance synthetic dataset throughout iterations')
    
axd['a'].plot(iterations,B_x_diff_mean_it, linewidth=1)
axd['a'].plot(iterations,B_x_diff_abs_mean_it, linewidth=1)
axd['a'].set_xlabel('iterations')
axd['a'].set_ylabel('error calculated intensity [$\mu$T]')
axd['a'].legend([' $\Delta_{int}$', '|$\Delta|_{int}$'])
axd['a'].annotate('a.', (0.95, 0.02), xycoords='axes fraction' , fontsize=10)
axd['a'].hlines(y=0, xmin=0-(q/20), xmax=q+(q/20), linewidth=1, color='k', linestyles='dashed')
axd['a'].set_xlim(0-(q/20), q+(q/20))

axd['b'].plot(iterations, X_euclean_it, linewidth=1)
axd['b'].plot(iterations, Y_euclean_it, linewidth=1)
axd['b'].set_ylim(-0.0001, 0.0009)
axd['b'].set_xlabel('iterations')
axd['b'].set_ylabel('variance [-]')
axd['b'].legend(['$NRM_{\epsilon}$', '$ARM_{\epsilon}$'])
axd['b'].annotate('b.', (0.95, 0.02), xycoords='axes fraction' , fontsize=10)
axd['b'].hlines(y=0, xmin=0-(q/20), xmax=q+(q/20), linewidth=1, color='k', linestyles='dashed')
axd['b'].set_xlim(0-(q/20), q+(q/20))

plt.show()

#%% calculate end-members, with optimal iteration, and calculate paleointensity with this optimal iteration

X = X_dataset
Y = Y_dataset
B_x_ref = B_x_ref_dataset
n = n_dataset

# make x and y matrix not devided by field strength
B_x_ref_matrix = np.zeros((n,n))
np.fill_diagonal(B_x_ref_matrix, B_x_ref)
x = np.matmul(B_x_ref_matrix, X)
x = np.matrix.transpose(x)
y = B_lab * Y
y = np.matrix.transpose(y)

# initial guess
S_x_old = np.zeros((m, k_x))
S_y_old = np.zeros((m, k_y))    
for i in range (m):    
    S_x_old[i, :] = np.linspace((0.05 + (i * 0.001)), 0, k_x)
    S_y_old[i, :] = np.linspace(0, (0.025 + (i * 0.001)) , k_y)
    
A_old = np.ones((n, m))
A_old = A_old / np.sum(A_old, 1).reshape(n,1)

# start iteration
for i in range (int(B_x_min_iteration)):
    
    A = alpha_1 * np.multiply(A_old, (np.matmul(Y, np.transpose(S_y_old))
        / (np.matmul(np.matmul(A_old, S_y_old), np.transpose(S_y_old)) + eps))) \
    + alpha_2 * np.multiply(A_old, (np.matmul(X, np.transpose(S_x_old))
        / (np.matmul(np.matmul(A_old, S_x_old), np.transpose(S_x_old)) + eps)))

    # calculate new E
    S_x = np.multiply(S_x_old, (np.matmul(np.transpose(A_old), X) / (np.matmul(np.matmul(np.transpose(A_old), A_old), S_x_old) + eps)))
    S_y = np.multiply(S_y_old, (np.matmul(np.transpose(A_old), Y) / (np.matmul(np.matmul(np.transpose(A_old), A_old), S_y_old) + eps)))
    
    # calculate the error percentage 
    X_check = np.matmul(A, S_x)
    Y_check = np.matmul(A, S_y)
    X_diff_mean_percentage = np.mean(np.absolute(X_check - X)) / np.mean(X) * 100
    Y_diff_mean_percentage = np.mean(np.absolute(Y_check - Y)) / np.mean(Y) * 100

    # the new end-members become the old end-members
    S_x_old = S_x
    S_y_old = S_y
    A_old = A / np.sum(A, 1).reshape(n,1)
    
    # boundary conditions
    for j in range (m):
        S_x_old[j, -1] = 0
        S_y_old[j, 0] = 0
        S_y_old[j, -1] = 0.025
        
        for o in range (k_x - 1): 
            if S_x_old[j, k_x-o-2] < S_x_old[j, k_x-o-1]: 
                S_x_old[j, k_x-o-2] = S_x_old[j, k_x-o-1] 
                 
        for o in range (k_y - 1): 
            if S_y_old[j, -2-o] > S_y_old[j, -1-o]: 
                S_y_old[j, -2-o] = S_y_old[j, -1-o] 
                
# calculation of paleointensity with the calculated end-members

# transpose the end-members for the calculation of paleointensity
s_x = np.transpose(S_x)
s_y = np.transpose(S_y)
s_y = s_y * B_lab
    
# allocate space for matrix a and B_x
a = np.zeros((m, n))
B_x = np.zeros(n)
    
# calculate paleointensity, done seperately for every measurement
B_x_seperate = np.zeros((n,m))
B_x_seperate_NaN = np.zeros((n,m))
for j in range (n):
    # calculate B_x from A, E_x and X
    a[:,j] = nnls(s_y, y[:, j].flat)[0]
    B_x[j] = np.linalg.lstsq( nnls(s_y, y[:, j].flat)[0].reshape(m,1), nnls(s_x, x[:, j].flat)[0].reshape(m,1), rcond=None)[0]

# transpose back
a = np.transpose(a)
s_x = np.transpose(s_x)
s_y = np.transpose(s_y)
s_y = s_y / B_lab
x = np.transpose(x)
y = np.transpose(y)

# compare the results to the known paleointensity of every measurement 
B_x = B_x.reshape(n,1)
B_x_diff = B_x - B_x_ref
B_x_diff_min = B_x_diff.min()
B_x_diff_max = B_x_diff.max()
B_x_diff_mean = np.mean(B_x_diff)
B_x_diff_pos = np.absolute(B_x_diff)
B_x_diff_abs_mean = np.mean(B_x_diff_pos)

# Plot the difference and absolute difference in paleointensity
print('Difference in paleointensity:', B_x_diff_mean)
print ('Absolute difference in paleointensity:', B_x_diff_abs_mean)

# plot the optimalization 
fig, axd = plt.subplot_mosaic([['a', 'b']], constrained_layout = True, figsize=(7.2,3), dpi=1200)
   
for i in range (m): 
    axd['a'].plot(x_as_NRM, np.array(S_x[i,:]).flatten())
    axd['b'].plot(x_as_ARM, np.array(S_y[i,:]).flatten())

axd['a'].set_title(f'{m} end-members: NRM demagnetization')
axd['b'].set_title(f'{m} end-members: ARM acquisition')
axd['a'].set_xlabel('Demagnetization steps [mT]')
axd['a'].set_ylabel('Normalized magnetization')
axd['b'].set_xlabel('Remagnetization steps [mT]')
axd['b'].set_ylabel('Normalized magnetization')
plt.show()

# Plot distribution of B_x_diff, the difference in calculated paleointensity compared to the reference paleointensity
plt.figure(figsize=(3.5, 2.625), dpi=1200)
plt.hist(B_x_diff, bins=40)
#plt.title('Distribution of |$\Delta_{int}$| per sample')
plt.xlabel('|$\Delta|_{int}$ [$\mu$T]')
plt.ylabel('amount of samples [-]')
plt.show()

#Write the results of the calculation to an exel file, this is formated to match the excel file used for the input data 
if which_data_set == "Data":
    
    workbook = xlsxwriter.Workbook('output_synthetic.xlsx')
    worksheet = workbook.add_worksheet()
    for i in range (n):
        worksheet.write(0, 0, 'B_x')
        worksheet.write(steps[i]+1, 0, B_x[i])
        worksheet.write(0, 1, 'B_x_diff')
        worksheet.write(steps[i]+1, 1, B_x_diff[i])
        for j in range (m):
            worksheet.write(0, 3+j, f'A {j+1}')
            worksheet.write(steps[i]+1, 3+j, A[i,j])
        for j in range (m):
            worksheet.write(0, 4+m+j, f'a {j+1}')
            worksheet.write(steps[i]+1, 4+m+j, a[i,j])
 
    workbook.close()

#%%