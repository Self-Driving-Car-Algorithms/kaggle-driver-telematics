# Kaggle Telematics

My solutiom to Kaggle's Axa Telematics challenge, 533rd place.

## Summary

We derive a set of features from original data for driver matching. A complimentary and effective 
approach would be to reduce the noise from the original data using an algorithm like
[Ramer-Douglas-Peucker](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm) and
then do trip matching.
Unfortunately, I only learned about that after the competition was over.

## Preprocessing

First we create the following set of features for each trip present on the raw data:
- distance - total trip distance
- points - number of measure points
- mean_inst_speed - mean instant speed
- sd_inst_speed - standard deviation of instant speed
- mean_avg_speed - mean average speed of trip
- sd_avg_speed - standard deviation of average speed 
- mean_acceleration - mean acceleration
- sd_acceleration - standard acceleration
- final_angle - angle between the vector formed by drawing a line from origin to the last data point and
		the x axis.

## Prediction

Several algorithms were tried but we settled on Extra-trees Regressor from Scikit-Learn.

## Running

Before running you should install the following dependencies:
- Python 2.7+
- NumPy
- Scikit-Learn

You can get the raw data from [Driver Telematics Analysis](https://www.kaggle.com/c/axa-driver-telematics-analysis/data).

After having all dependencies and raw data, just run ./pipeline.sh for running all prediction pipeline. It will take a few
hours.

