### Few_Nerds

# train_demo.py
python3 train_demo.py --N 2 --K 1 --trainN 2 --use_sampled_data --train_iter 1600 ### data/episodes-data/inter/train_n_k.jsonl, test..., dev...


# run_supervised.py

python3 run_supervised.py --data_dir sup --model_type bert --model_name_or_path bert-base-uncased --output_dir supervised_output --do_train --do_eval --evaluate_during_training  --logging_steps 20 --save_steps 500000 --labels data_conll/labels.txt --do_lower_case --overwrite_cache --overwrite_output_dir


# merge_and_convert_to_supervised.py

python3 merge_and_convert_to_supervised.py train_art_2_1.jsonl test_art_2_1.jsonl --output_dir


# scores_plot.py 
python scores_plot.py "input.txt"
