
for shortcut_nums in 0 1 2 3 4 5 6 0 1 2 3 4 5 6 0 1 2 3 4 5 6
do
    python main_v4.py --depth=5 --num_seg=10 --width=6 --epochs=100 --shortcut_num=$shortcut_nums
done



