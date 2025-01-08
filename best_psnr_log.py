import re

# Parse log file to find epoch, iteration, and psnr values
psnr_data = []
pattern = r'<epoch:(\d+), iter:\s+([\d,]+)> psnr: ([\d.e+-]+)'

log_contents = "../SR3/SR3_wdTest/experiments/s14_ori/logs/val.log"
with open(log_contents, 'r') as f:
    log_contents = f.read()

# Finding all matches in the log content
matches = re.findall(pattern, log_contents)
for i, match in enumerate(matches):
    epoch = int(match[0])
    iteration = int(match[1].replace(',', ''))  # Remove commas for parsing as integer
    psnr = float(match[2])
    val_name = "val1" if i % 2 == 0 else "val2"
    psnr_data.append((val_name, epoch, iteration, psnr))

# print(psnr_data)
# Find the epoch with the highest PSNR
max_psnr_entry = sorted(psnr_data, key=lambda x: x[3], reverse=True)


for i in range(100):
    if max_psnr_entry[i][2] < 100000 and max_psnr_entry[i][0] == "val1":
        print(f"Validation: {max_psnr_entry[i][0]}, Iteration: {max_psnr_entry[i][2]}, PSNR: {max_psnr_entry[i][3]}")
        break

for i in range(len(max_psnr_entry)):
    if max_psnr_entry[i][2] < 100000 and max_psnr_entry[i][0] == "val2":
        print(f"Validation: {max_psnr_entry[i][0]}, Iteration: {max_psnr_entry[i][2]}, PSNR: {max_psnr_entry[i][3]}")
        break
