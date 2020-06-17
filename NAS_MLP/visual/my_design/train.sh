for num_segs in 20
do
    for depths in 20
    do
        for shortcut_nums in 0 171
        do

            python main.py --depth=$depths --num_seg=$num_segs --width=16 --shortcut_num=$shortcut_nums
        done
    done
done

