



for num_segs in 4 8
do
    for depths in 6 8
    do
        for shortcut_nums in 10 15 20
        do

            python main_v4.py --depth=$depths --num_seg=$num_segs  --shortcut_num=$shortcut_nums
        done
    done
done

for num_segs in 4 8
do
    for depths in 9 11
    do
        for shortcut_nums in 20 30 40
        do

            python main_v4.py --depth=$depths --num_seg=$num_segs  --shortcut_num=$shortcut_nums
        done
    done
done


for num_segs in 4 8 16
do
    for depths in 16 20
    do
        for shortcut_nums in 100 135 170 
        do
            python main_v4.py --depth=$depths --num_seg=$num_segs  --shortcut_num=$shortcut_nums --width=16
        done
    done
done