import re

# Parse log file to find epoch, iteration, and psnr values
psnr_data = []
pattern = r'<epoch:(\d+), iter:\s+([\d,]+)> psnr: ([\d.e+-]+)'

log_contents = "../SR3/SR3_wdTest/experiments/wdtest_16_64_241107_063704/logs/val.log"
with open(log_contents, 'r') as f:
    log_contents = f.read()

# Finding all matches in the log content
matches = re.findall(pattern, log_contents)
for match in matches:
    epoch = int(match[0])
    iteration = int(match[1].replace(',', ''))  # Remove commas for parsing as integer
    psnr = float(match[2])
    psnr_data.append((epoch, iteration, psnr))

# Find the epoch with the highest PSNR
max_psnr_entry = max(psnr_data, key=lambda x: x[2])
print("Epoch with the highest PSNR: ", max_psnr_entry)
