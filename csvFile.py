# importing csv module 
import csv 
import json

with open(".../data_file.json", "r") as read_file:
    data = json.load(read_file)


# csv file name 
filename = ".../results.csv"

# initializing the titles and rows list 
fields = [] 
rows = [] 

# reading csv file 
with open(filename, 'r') as csvfile: 
	# creating a csv reader object 
	csvreader = csv.reader(csvfile) 
	
	# extracting field names through first row 
	#fields = csvreader.next()
	fields = next(csvreader)

	# extracting each data row one by one 
	for row in csvreader: 
		rows.append(row) 

	# get total number of rows 

rouge1=[]
for row in rows: 
    # parsing each column of a row
    if row !=[] and row[0]=="ROUGE-1":
        rouge1.append((float(row[3]),float(row[4]),float(row[5]),row[1]))
r=[]
p=[]
f=[]
for i,j,k,l in rouge1:
    r.append(i*(1-(max(max(240-data[l],data[l]-250),0))/240))  
    p.append(j*(1-(max(max(240-data[l],data[l]-250),0))/240))
    f.append(k*(1-(max(max(240-data[l],data[l]-250),0))/240))

print ("===================Average(rouge1)==========================")
print ("Rouge1.details=",rouge1)
print ("\nAvg.Recall=",sum(r)/10.0)
print ("Avg.Precision=",sum(p)/10.0)
print ("Avg.F-Score=",sum(f)/10.0)
rouge2=[]
for row in rows: 
    # parsing each column of a row
    if row !=[] and row[0]=="ROUGE-2":
        rouge2.append((float(row[3]),float(row[4]),float(row[5]),row[1]))
r=[]
p=[]
f=[]
for i,j,k,l in rouge2:
    r.append(i*(1-(max(max(240-data[l],data[l]-250),0))/240))  
    p.append(j*(1-(max(max(240-data[l],data[l]-250),0))/240))
    f.append(k*(1-(max(max(240-data[l],data[l]-250),0))/240))

print ("===================Average(rouge2)==========================")
print ("Rouge2.details=",rouge2)
print ("\nAvg.Recall=",sum(r)/10.0)
print ("Avg.Precision=",sum(p)/10.0)
print ("Avg.F-Score=",sum(f)/10.0)

rougeSU4=[]
for row in rows: 
    # parsing each column of a row
    if row !=[] and row[0]=="ROUGE-SU4":
        rougeSU4.append((float(row[3]),float(row[4]),float(row[5]),row[1]))
r=[]
p=[]
f=[]
for i,j,k,l in rougeSU4:
    r.append(i*(1-(max(max(240-data[l],data[l]-250),0))/240))  
    p.append(j*(1-(max(max(240-data[l],data[l]-250),0))/240))
    f.append(k*(1-(max(max(240-data[l],data[l]-250),0))/240))
print ("===================Average(rougeSU4)==========================")
print ("RougeSU4.details=",rougeSU4)
print ("\nAvg.Recall=",sum(r)/10.0)
print ("Avg.Precision=",sum(p)/10.0)
print ("Avg.F-Score=",sum(f)/10.0)

