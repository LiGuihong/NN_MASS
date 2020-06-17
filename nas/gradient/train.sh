# params: depth  num_seg  shortcut_num

for num_segs in 4
do
    for depths in 5
    do
        for shortcut_nums in 0 1 2 3 4 5
	do
            python main_v4.py --depth=$depths --num_seg=$num_segs  --shortcut_num=$shortcut_nums
        done
    done
done
