



for num_segs in 8 12 16
do
    for depths in 6
    do
        for shortcut_nums in 4 6 8 10
        do

            python main_v5.py --depth=$depths --num_seg=$num_segs  --shortcut_num=$shortcut_nums
        done
    done
done


for num_segs in 8 12 16
do
    for depths in 8
    do
        for shortcut_nums in 10 12 14 16 18 20
        do

            python main_v5.py --depth=$depths --num_seg=$num_segs  --shortcut_num=$shortcut_nums
        done
    done
done

for num_segs in 8 12 16
do
    for depths in 10
    do
        for shortcut_nums in 10 15 20 25 30 35 40
        do

            python main_v5.py --depth=$depths --num_seg=$num_segs  --shortcut_num=$shortcut_nums
        done
    done
done

