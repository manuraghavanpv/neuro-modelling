import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

#QUESTION-1: Here, I used the following code to obtain the pRF and generate the predicted fMRI time-series

#Step-1: Loading the images and inspecting the shape of the given matrix
data = np.load('/Users/manu/Desktop/NiN Project/pythonProject/transfer_2777608_files_b560e89e/images.npz')
images = data['ims']
print(f"Shape of the 3D matrix: {images.shape}")

#Given-1: Matrix has the dimensions visual space x visual space x time (ie, 2D visual stimulus over time).
#Given-2: The visual space of both horizontal and vertical are the same (since 10 x 10 is given).
#Assumption-1: Therefore, a square matrix for visual space is assumed when defining parameters in Step-2

#Step-2: Defining the parameters
visual_space = images.shape[0]  #See Assumption-1
eccentricity = 2  #given in the question
sigma = 1  #given in the question

#Step-3: Checking the range of the stimulus values and normalising it
print(f"Stimulus min value: {images.min()}")
print(f"Stimulus max value: {images.max()}")
#From this, it appeared that the minimum value of stimulus is 1 (background) and maximum is 254 (active stimulus)
#Therefore, this was further normalised to make sure that background corresponds to 0 and stimulus corresponds to 1
images = (images - images.min()) / (images.max() - images.min())

#Step-4: Calculating x and y co-ordinates of pRF (redundant but shown to explain thought process)
#I tried this initially : "x_pRF = visual_space// 2 + eccentricity" but it failed to work since "0" was coming up in the division formula of pRF.
#I reasoned that this is probably due to a couple of things: (1) eccentricity not being converted from degrees to the corresponding index in the visual space, (2) Gaussian pRF calculation not being centered correctly.
#x-co-ordinate for the center of pRF = center of matrix + offset by 2 degrees of eccentricity
#y-co-ordinate for the center of pRF = center of matrix
x_pRF = visual_space//2 + int(eccentricity * (visual_space / 10)) #eccentricity was converted from degrees to the corresponding index
y_pRF = visual_space//2

#Step-5: Creating a 2D grid: Here, x and y represent spacees from -5 to +5 degrees and the meshgrid function of numpy was used to create a 2D grid across the visual space
x = np.linspace(-5, 5, visual_space)
y = np.linspace(-5, 5, visual_space)
X, Y = np.meshgrid(x, y)

#Step-6: Calculating the Gaussian pRF
#Initially, I mistakenly used the pRF formula directly from the standard model (Dumolin and Wandel, 2008): prf = np.exp(-((X - (x_pRF - visual_space // 2))**2 + (Y - (y_pRF - visual_space // 2))**2) / (2 * sigma**2))
#However, with this, the sum of pRF was coming to zero as it was incorrectly centered.
#The mistake: The original formula translates the indices back to a central origin in degrees which is redundant as X and Y already represent degrees from -5 to +5
#From my understanding, the Gaussian function is calculated based on the distance from center point (eccentricity for x and 0 for y) and given spread (sigma)
#Therefore, I made the following changes to the formula to directly correspond to X and Y in the grid and create a 2D Gaussian centered at specified eccentricity.
prf = np.exp(-((X - (eccentricity))**2 + (Y)**2) / (2 * sigma**2))

#Step-7: Normalising the pRF
prf /= prf.sum()

#Step-8: Visualising the pRF
plt.imshow(prf, extent=(-5, 5, -5, 5), origin='lower', cmap='hot')
plt.colorbar()
plt.title('2D Gaussian pRF')
plt.xlabel('Angle(degrees)')
plt.ylabel('Angle(degrees)')
plt.show()

#Step-9: Creating the predicted time-series
images = 1-images #Initially, I was getting an inverted time series. So I did this to ensure the correct representation of fMRI signal with respect to stimulus
#Initialising:
predicted_time_series = np.zeros(images.shape[-1])
#Now, for each point in time, the dot product of pRF and image slice is used to generate the predicted time series.
for t in range(images.shape[-1]):
    predicted_time_series[t] = np.sum(prf * images[:, :, t])

#Step-10: Correcting for time-interval
#Since it was given that every time point was separated by 1.5s in given data, the following was done to ensure that the predicted time-series has a time interval of 1.5s (ie, aligned with data acquisition)
time_points = np.arange(0, len(predicted_time_series) * 1.5, 1.5) #An array of time points starting from 0 with an interval of 1.5

#Step-11: Visualising the predicted time-series
plt.plot(time_points, predicted_time_series)
plt.xlabel('Time(s)')
plt.ylabel('Predicted fMRI Signal')
plt.title('Predicted Time-Series')
plt.show()

#This provided a Gaussian pRF and the predicted time series where peaks correspond to high overlap between the pRF and the stimulus

#QUESTION-2: Here I used the following code to demonstrate depth effects across cortical layers where superficial/outer layers have higher intensity and lower layers have lower intensity

#Step-1: Normalising predicted time series
predicted_time_series = (predicted_time_series - np.min(predicted_time_series)) / (np.max(predicted_time_series) - np.min(predicted_time_series))
#Initially, I didn't normalise and got constant values as outputs and so I decided to normalise the predicted time series values between 0 and 1
#This normalisation provided a standardised baseline, dynamic range, consistent scaling and accurate smoothing in the following steps

#Simulating the time-series for different cortical depths with depth being assumed to be inversely related to BOLD signal
outer = predicted_time_series * 1.5 #Higher amplitude for superficial layers (higher BOLD response due to higher vascular density)
middle = predicted_time_series * 1.0 #Moderate amplitude for middle layer (intermediate vascular density)
deep = predicted_time_series * 0.5 #Lower amplitude for deep layers (lower vascular density)

#A simple moving-average kernel was used to smoothen out the deeper layers in order to make the BOLD response realistic
deep = np.convolve(deep, np.ones(5)/5, mode='same')

# Generating the time points (with a 1.5s TR as mentioned before)
time_points = np.arange(0, len(predicted_time_series) * 1.5, 1.5)

# Plotting the time-series to demonstrate changes with cortical depth
plt.figure(figsize=(10, 6))
plt.plot(time_points, outer, label='Outer Layers')
plt.plot(time_points, middle, label='Middle Layers')
plt.plot(time_points, deep, label='Deep Layers')
plt.xlabel('Time (seconds)')
plt.ylabel('Predicted fMRI Signal')
plt.title('Predicted fMRI Time-Series Across Different Cortical Layers')
plt.legend()
plt.show()

#This provided a time series which demonstrates the difference in fMRI signal across three cortical layers of varying depth