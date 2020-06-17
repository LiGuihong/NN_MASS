# params: depth  num_seg  shortcut_num

for num_segs in 2 3 4 5 6 8 10 20
do
    for depths in 3 4 5 6 7 8 9 10 12
    do
        for shortcut_nums in 10 20 30 40 60 80 100 120 160 200
        do
            for idx in 1 2 3 4 5
            do
                python main_v2.py --depth=$depths --num_seg=$num_segs  --shortcut_num=$shortcut_nums
            done
        done
    done
done