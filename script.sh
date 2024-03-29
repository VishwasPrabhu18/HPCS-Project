#!/bin/bash

# Compile the program
g++ serial.cpp -o output `pkg-config --cflags --libs opencv4`

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed. Exiting."
    exit 1
fi

# Set the directory containing image files
IMAGE_DIRECTORY="new/"

# Set the number of repetitions
REPEATS=5

# Iterate through all image files in the directory
for image_file in $IMAGE_DIRECTORY*.jpg; do
    # Check if the file is a regular file
    if [ -f "$image_file" ]; then
        total_duration=0
        # Print information before running the program
        echo "Running ./output for $image_file $REPEATS times"

        # Run your program and calculate the average duration
        for ((i=1; i<=$REPEATS; i++)); do
            duration=$(./output "$image_file" | grep "3DLBP descriptor calculation time" | awk '{print $5}')
            echo "3DLBP descriptor calculation time: $duration ms"

            # Add the duration to the total
            total_duration=$((total_duration + duration))

            # Print a separator line for better readability
            echo "------------------------------------------"
        done

        # Calculate the average duration
        average_duration=$((total_duration / REPEATS))
        echo -e "Average 3DLBP descriptor calculation time: $average_duration ms\n"
    fi
done
