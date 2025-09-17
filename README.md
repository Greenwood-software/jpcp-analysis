# jpcp-analysis
Jointed Plain Concrete Pavement Analysis from TSD data

#### Author(s): David Malmgren-Hansen 

## DATA: example_data.csv
PLEASE SEE NOTICE ABOUT THIS DATA

The File contains TSD measurements conducted by Greenwood Engineering A/S in 2020, in nothern Germany. It is part of a study that was published:

Nielsen, Christoffer P., Mahdi Rahimi Nahoujy, and Dirk Jansen. "Measuring joint movement on rigid pavements using the traffic speed Deflectometer." Journal of Transportation Engineering, Part B: Pavements 149.2 (2023): 04023002.

The measurement is outdated and does not represent the current conditions of any roads and should be used as example or test purpose for the code provided here.

The file does not show the full extent of possible sensor data collection possible with Greenwood TSDs and is a subset for the specific purpose in this code. 

## joint_analysis.py

Calculate effective joint spring constant

positional arguments:
  path                  Path to folder or filename with exports. If left empty it plots an example.

options:
  -h, --help            show this help message and exit
  -i INTERVAL INTERVAL, --interval INTERVAL INTERVAL
                        e.g. "-i 340 1005" interval to average over



