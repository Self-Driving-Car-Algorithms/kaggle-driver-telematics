#
# preprocess.py
#
# Preprocess data and generate a file with trip statistics for each driver.

from math import fabs, sqrt, atan2, degrees, fsum
from functools import partial
from os import listdir, path
from sys import argv, stdout
import numpy as np
import pandas as pd
import time

from util import edist, projection, sd

def read_lines(data):
    lines = []

    for r in data.iterrows():
        n, tpl = r
        lines.append((tpl[0], tpl[1]))

    return lines

def norm(v):
    return sqrt(v[0]**2 + v[1]**2)

def vector_sum(a, b):
    return (a[0] + b[0], a[1] + b[1])

def pair_distance(iterable):
    return [edist(iterable[i], iterable[i-1]) for i in range(1, len(iterable))]

def read_trip(data):
    """
    The main loop is ugly, I know, but I'm doing everything in the same place for performance reasons.
    """
    vectors = read_lines(data)
    stats = {}
    distances = []
    accelerations = []

    roundto1 = partial(round, ndigits=1)
    total_distance = 0.0
    resultant_vector = (0, 0)
    total_acceleration = 0.0
    total_average_speed = 0.0

    for i in range(1, len(vectors)):
        distance = roundto1(edist(vectors[i], vectors[i-1]))
        distances.append(distance)
        total_distance += distance
        resultant_vector = (resultant_vector[0] + vectors[i][0], resultant_vector[1] + vectors[i][1])
        average_speed = total_distance / i
        total_average_speed += average_speed
        acceleration = distance - distances[-1]
        accelerations.append(acceleration)
        total_acceleration += acceleration

    # Drop origin.
    #vectors = vectors[1:]
    n = len(vectors)

    # ?
    accelerations = [0] + accelerations

    # Calculate the angle in degrees between the final vector and the x-axis. Use atan2
    # because it knows about components signal and thus generates correct angles.
    # PS. We gotta have a better way to do this.
    final_angle = np.digitize([degrees(atan2(resultant_vector[1], resultant_vector[0]))], np.linspace(0, 360, 13))[0]
    
    mean_inst_speed = total_distance / n
    sd_inst_speed = sd(distances, mean_inst_speed)

    mean_average_speed = total_average_speed / n
    sd_average_speed = sd(distances, mean_average_speed)
    
    mean_acceleration = total_acceleration / n
    sd_acceleration = sd(accelerations, mean_acceleration)

    # Bin and normalize instant speeds frequencies to transform them into feature vector.
    distances = np.array(distances)
    bins = np.linspace(distances.min(), distances.max(), 80)
    digitized = np.digitize(distances, bins)
    counts = np.bincount(digitized)
    spread = float(counts.max() - counts.min())
    counts = (counts - counts.min()) / spread

    for n, d in enumerate(counts):
        s = 'speed%d' % (n)
        stats[s] = d
   
    stats['distance'] = total_distance
    stats['points'] = n
    stats['mean_inst_speed'] = mean_inst_speed
    stats['sd_inst_speed'] = sd_inst_speed
    stats['mean_avg_speed'] = mean_average_speed
    stats['sd_avg_speed'] = sd_average_speed
    stats['mean_acceleration'] = mean_acceleration
    stats['sd_acceleration'] = sd_acceleration
    stats['final_angle'] = final_angle
    # WARNUNG: experimental!
    #stats['vars1'] = var_inst_speed + var_average_speed
    #stats['vars2'] = var_inst_speed + var_acceleration
    #stats['vars3'] = var_average_speed + var_acceleration
    #stats['vars4'] = var_inst_speed + var_average_speed + var_acceleration

    return stats

#def read_trip(data):
#    stats = {}
#    
#    vectors = read_lines(data)
#
#    # First read all points as vectors from the origin.
#
#    roundto1 = partial(round, ndigits=1)
#    distances = np.array(map(roundto1, pair_distance(vectors)))
#    
#    # Drop origin.
#    vectors = vectors[1:]
#    n = len(vectors)
#
#    #total_distance = fsum(norms)
#    total_distance = fsum(distances)
#
#    resultant_vector = reduce(vector_sum, vectors, (0, 0))
#    # Calculate the angle in degrees between the final vector and the x-axis. Use atan2
#    # because it knows about components signal and thus generates correct angles.
#    # PS. We gotta have a better way to do this.
#    final_angle = np.digitize([degrees(atan2(resultant_vector[1], resultant_vector[0]))], np.linspace(0, 360, 13))[0]
#    
#    # Each point is one second apart in time from the other, so essencially we have velocities.
#    #instant_speed = norms
#    instant_speeds = distances
#
#    # Take average velocities.
#    #average_speeds = [(norms[i - 1] / i) for i in range(1, n + 1)]
#    average_speeds = [(distances[i - 1] / i) for i in range(1, n + 1)]
#
#    #accelerations = [norms[i] - norms[i-1] for i in range(1, len(norms))] + [0]
#    accelerations = [0] + [distances[i] - distances[i - 1] for i in range(1, len(distances))]
#
#    mean_inst_speed = total_distance / n
#    sd_inst_speed = sd(distances, mean_inst_speed)
#
#    mean_average_speed = fsum(average_speeds) / n
#    sd_average_speed = sd(distances, mean_average_speed)
#    
#    mean_acceleration = fsum(accelerations) / n
#    sd_acceleration = sd(accelerations, mean_acceleration)
#
#    # Bin and normalize instant speeds frequencies to transform them into feature vector.
#    bins = np.linspace(instant_speeds.min(), instant_speeds.max(), 80)
#    digitized = np.digitize(instant_speeds, bins)
#    counts = np.bincount(digitized)
#    spread = float(counts.max() - counts.min())
#    counts = (counts - counts.min()) / spread
#
#    for n, d in enumerate(counts):
#        s = 'speed%d'%(n)
#        stats[s] = d
#   
#    stats['distance'] = total_distance
#    stats['points'] = n
#    stats['mean_inst_speed'] = mean_inst_speed
#    stats['sd_inst_speed'] = sd_inst_speed
#    stats['mean_avg_speed'] = mean_average_speed
#    stats['sd_avg_speed'] = sd_average_speed
#    stats['mean_acceleration'] = mean_acceleration
#    stats['sd_acceleration'] = sd_acceleration
#    stats['final_angle'] = final_angle
#    # WARNUNG: experimental!
#    #stats['vars1'] = var_inst_speed + var_average_speed
#    #stats['vars2'] = var_inst_speed + var_acceleration
#    #stats['vars3'] = var_average_speed + var_acceleration
#    #stats['vars4'] = var_inst_speed + var_average_speed + var_acceleration
#
#    return stats

def read_driver_trips(filepath, driver):
    """
    Read all trips of one driver (directory).
    """
    #files = listdir(path.join(filepath, driver_dir))

    #trip_stats = [read_trip(path.join("drivers", driver_dir), trip) for trip in files]
    
    p = path.join(data_path, driver)
    df = pd.read_pickle(p)
    trip_stats = []

    trip_range = range(1,201)

    for t in trip_range:
        #print "Reading", t
        delta = time.time()
        stats = read_trip(df[df['trip'] == t])
        #print str(time.time() - delta)
        stats['driver'] = driver
        stats['trip'] = str(t)
        trip_stats.append(stats)
    
    return trip_stats

def main(data_path, rng):
    allfiles = listdir(data_path)
    # Just in case.
    allfiles = sorted(allfiles)
    
    if rng:
        low, high = rng.split("-")
        low = int(low)
        high = int(high)
        howmany = high - low 
    else:
        l = len(allfiles)
        low, high = 0, l
        howmany = high - low
    
    n = 0
    
    print "Preprocessing features, files in the range %s"%(rng)
    print "Progress:"
    # Read each driver trips
    for i in range(low, high):
        driver = allfiles[i]
        #output = open("data/stats/%s"%(driver), "w")
        trips = read_driver_trips(data_path, driver)
        
        #headers = trips[0].keys()
        #output.write(','.join(headers) + '\n')
        df = pd.DataFrame(trips)
        outputname = driver.split('.')[0]
        pd.to_pickle(df, "data/stats/%s.pkl"%(outputname))

        if n%10 == 0:
            print "%d more to go"%(howmany-n)

        n += 1
        
if __name__ == "__main__":
    data_path = "data/aggregated"
    rng = None

    if len(argv) > 1:
        if argv[1] == "-r":
            rng = argv[2]

    main(data_path, rng)

