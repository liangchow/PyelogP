from pyelogp import Data

# Example 1 --- Run analysis on .csv file.
# Becker et al. (1987): Wallaceburg Clay, P'c (Casangrade)=115 kPa, P'vy=120 kPa, OCR=1.33
# See data/example.csv
d1 = Data.from_csv("data/example.csv", e0=1.24)
result = d1.find_pc()

print("Preconsolidation pressure (Pc):", result.pc)
print("Void ratio at pc (e_pc):", result.e_pc)
print("R² segment 1:", result.r2_seg1)
print("R² segment 2:", result.r2_seg2)


# Example 2 --- Run analysis directly on pressure and void_ratio arrays.
# TPM (1996): Louiseville Clay, P'c=165 kPa, OCR=2.8
pressure = [59, 90, 120, 150, 165, 172, 184, 222, 300, 400]
void_ratio = [2.115, 2.113, 2.098, 2.083, 2.055, 2, 1.8, 1.5, 1.3, 1.193]
e0=1.21

d2 = Data(pressure=pressure, void_ratio=void_ratio, e0=e0)
result = d2.find_pc()

print("Preconsolidation pressure (Pc):", result.pc)
print("Void ratio at pc (e_pc):", result.e_pc)
print("R² segment 1:", result.r2_seg1)
print("R² segment 2:", result.r2_seg2)

# returns:
# Wallaceburg Clay:
# Pc = 113.86
# e_pc = 1.0804
# R2,seg1 = 0.983665
# R2,seg2 = 0.998988

# Louiseville Clay:
# Pc = 164.49
# e_pc = 2.0573
# R2,seg1 = 0.89377
# R2,seg2 = 0.96937
