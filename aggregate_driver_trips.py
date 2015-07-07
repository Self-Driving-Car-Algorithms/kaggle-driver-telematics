import sys
import os
import numpy as np
import pandas as pd

def aggregate(rng=None):
    drivers = sorted([int(directory) for directory in os.listdir("drivers/")])

    low = 0
    high = len(drivers) - 1

    if rng:
        print "Executing on range", rng
        l, h = rng.split("-")

        if l != "":
            low = int(l)

        if h != "":
            high = int(h) 


    try:
        if len(drivers) != 2736:
            raise
    except:
        print "Error: %d drivers found instead of 2736"%(len(drivers))
        sys.exit(0)

    drivers = drivers[low:high]

    trips = range(1, 201)

    n = 0
    for driver in drivers:
        if n%100 == 0:
            print "n =", n

        df_driver = pd.DataFrame()

        for trip in trips:
            df = pd.read_csv("drivers/%s/%d.csv"%(driver, trip))
            df['trip'] = trip
            df_driver = pd.concat([df_driver, df], ignore_index=True)
        
        df_driver.to_pickle("data/aggregated/driver%s.pkl"%(driver))

        n += 1

if __name__ == '__main__':
    if len(sys.argv) > 2:
        if sys.argv[1] == "-r":
            aggregate(sys.argv[2])
    else:
        aggregate()


