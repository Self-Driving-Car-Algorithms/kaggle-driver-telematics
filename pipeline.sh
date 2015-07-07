#!/usr/bin/env sh

mkdir -p data

if [ ! -f data/aggregated ]
then
    echo "Aggregating driver trips..."
    mkdir -p data/aggregated
    time python aggregate_driver_trips.py
fi

echo "Trips aggregated"

if [ ! -f data/stats/ ]
then
    echo "Preprocessing data..."
    mkdir -p data/stats
    time python preprocess.py
fi

echo "Data preprocessed."

echo "Training Model..."
time python main.py

echo "Done."
