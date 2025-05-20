# -----STARTING YEAR-----
Ybeg = 2000

# ----END YEAR-----
Yend = 2009

# -----YEAR SPAN----
Tspan = (Yend - Ybeg) + 1

# -----DAY IN YEAR-----
# n.B.: To be updated with leap years
DinY = 365

# -----SECONDS IN DAY-----
SinD = 86400

# -----NUMBER OF FILES TO BE TREATED-----
nf = 2

# -----FILES STARTING AND END YEAR-----
chlfstart = [2000, 2005]
chlfend = [2004, 2009]

# -----THIS IS THE DATE FROM WHICH SECONDS START TO BE COUNTED-----
startingyear = 1981
startingmonth = 1
startingday = 1

LS = 0
LE = 0

# Initialize cumulative SZTtmp counter
total_SZTtmp = 0 

# -----YEAR SEQUENCE-----
ysec = list(range(Ybeg, Yend + 1))

# Define the number of days in each month (for non-leap years)
days_in_months_non_leap = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
days_in_months_leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]