DATA=../../data/SED_HS
cat $DATA/test_my.txt | while read line
do
	display $line
done