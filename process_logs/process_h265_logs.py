import sqlite3
import pandas as pd
import csv


data_folder="bair"
fom_file = data_folder + "/fom.csv"
power_file = data_folder+"/jtop.csv"
out_csv_file = data_folder+"/out.csv"

fom = pd.read_csv(fom_file, header=None)
power = pd.read_csv(power_file)

power_readings_per_file = []
for i, row in fom.iterrows():
    file_name,num_of_frames, wait_start, wait_end, start, end, src_size, kp_size, compressed_kp_size = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]
    idle_power_readings = power.query("{} < ts < {}".format(wait_start+1, wait_end-1))
    # chosen_idle_power_reading = idle_power_readings["total power cur"][0]
    power_readings = power.query("{} < ts < {}".format(start+1, end-1))
    total_power_readings = []
    for i, reading in power_readings.iterrows():
        total_power_readings.append(reading["total power cur"])
    avg_power_reading = sum(total_power_readings)/len(total_power_readings)/1000

    total_idle_power_readings = []
    for i, reading in idle_power_readings.iterrows():
        total_idle_power_readings.append(reading["total power cur"])
    avg_idle_power_reading = sum(total_idle_power_readings) / len(total_idle_power_readings) / 1000

    duration = end - start

    avg_conversion_power = avg_power_reading-avg_idle_power_reading

    total_energy = avg_conversion_power * duration
    current_data = {"file":file_name,"frames":num_of_frames, "start":start, "end":end,"src_file_size":src_size,"compressed_kp_size":compressed_kp_size/10, "duration(sec)":duration/10,"avg_total_idle_power_reading":avg_idle_power_reading, "avg_total_processing_power_reading(W)":avg_power_reading,"avg_conversion_power(W)":avg_conversion_power, "total_energy(J)":total_energy/10}

    power_readings_per_file.append(current_data)

# print(power_readings_per_file)

with open(out_csv_file, 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, power_readings_per_file[0].keys())
    dict_writer.writeheader()
    dict_writer.writerows(power_readings_per_file)

print("that's all folks")
