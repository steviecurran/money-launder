#!/bin/csh

set i = 1
set file = "N=10.csv"
set last = 101  # ADD ONE MORE THAN WANTED AS NO <=
set OUTFILE = "$file-$last-runs.csv"
#echo $OUTFILE
rm grid_runs/*.csv

while ($i < $last)
    echo "On loop $i of $last"
    ./DL-grid.py
    mv $file grid_runs/run_$i.csv
@ i = $i + 1 
end  

ls grid_runs/

sed -i -e 's/, /,/g' header.txt 
sed -i -e "s/\'//g" header.txt 
head -1 header.txt >  $OUTFILE
cat grid_runs/*.csv >> $OUTFILE
wc -l $OUTFILE
