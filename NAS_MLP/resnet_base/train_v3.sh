
for num_segs in 4 6 8 10
do
    for depths in 6
    do
        for shortcut_nums in 6 7 8 9 10
        do

            python main_v3.py --depth=$depths --num_seg=$num_segs  --shortcut_num=$shortcut_nums
        done
    done
done



for num_segs in 4 6 8 10
do
    for depths in 8
    do
        for shortcut_nums in 10 12 14 16 18 20
        do

            python main_v3.py --depth=$depths --num_seg=$num_segs  --shortcut_num=$shortcut_nums
        done
    done
done



for num_segs in 4 6 8 10
do
    for depths in 10
    do
        for shortcut_nums in 4 8 12 16 20
        do

            python main_v3.py --depth=$depths --num_seg=$num_segs  --shortcut_num=$shortcut_nums
        done
    done
done

for num_segs in 4 6 8 10
do
    for depths in 12
    do
        for shortcut_nums in 5 10 15 20 25
        do

            python main_v3.py --depth=$depths --num_seg=$num_segs  --shortcut_num=$shortcut_nums
        done
    done
done
