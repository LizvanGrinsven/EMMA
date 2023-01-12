Steps to use the EMMA method

The EMMA.py code is two fold, where during the first part the end-members are calculated from the dataset and in the second part these end-members are used to calculate the paleointensity for the same dataset to test the validity of the previously calculated end-members.

1. Make an input file with NRM demagnetization and ARM acquisition data in excel, an example can be found in the package called input_synthetic.xlsx data. As a default the input_synthetic.xlsx is used, which contains the laboratory magnetized samples used in the accompanying paper: An end-member modelling approach to pseudo-Thellier paleointensity data

2. Open the EMMA.py code and fill out the requested variables
	m = the desired number of end-members you wish to unmix the dataset
	H_lab = the lab field strength used during the ARM acquisition measurement
	
	then there are some options to preprocess the data
	removing outliers: is highly recommended, since a large range in maximum NRM values greatly disturbes the unmixing, even though they are only a small part of the dataset.
	removing overprint: recommend if using samples with a large overprint, this is not applicable for the lab magnetized samples from the input_synthetic dataset.
	removing locations: It is possible to remove certain parts of the dataset, based on the name given under locations in the input file
	
	alpha_1 and alpha 2 are weighting factors, which determine the weight for variable A toward NRM or ARM unmixing
	
	Lastly the name of the input file needs to be given, there is also a possibility to test a computer generated numerical dataset, in which case different percentage of noise can be added (note: the percentage given is the percentage of the norm of the entire dataset as an input for the standard deviation of the gaussian noise).

3. The EMMA.py code produces written tekst in the terminal, images and an excel file for output. 
	
	The values in the terminal display the number of samples in the unmixing scheme after preprocessing, optimal iteration gives the iteration at which the the unmixing scheme calculates the most accurate paleointensities, The (absolute) difference in paleointensity gives the average error for the calculated paleointensities with the calculated end-members.
	
	The first image displays the range of maximum NRM value (which are normalized by the maximum ARM value), and the dataset with and without the outliers, in case of choosing to remove the outliers. 
	The second image displays the error of the calculated end-members throughout the iterations
	The third image displays the calculated end-members
	The fourth image displays the distribution of the error of the calculated paleointensities
	
	Lastly the excel file displays the calculated paleointensities and the error in comparison with the known paleointensities. The excel also gives the upper case A which is the distribution variable for when the end-members are calculated for the optimal iteration and the lower case a which is the distribution variable for when the variable is calculated from the end-members and dataset to calculate the paleointensity.