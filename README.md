### Few_Nerds

# train_demo.py
python3 train_demo.py --N 1 --trainN 1 --K 1 --Q 1 --use_sampled_data --prefix combined --train_iter 2075 --val_step 10 --val_iter 450 --output_file_name combined_1_1.txt


# run_supervised.py

python run_supervised.py --data_dir supervised/data_supervised/combined/combined_1_1 --model_type bert --model_name_or_path bert-base-uncased --output_dir supervised_output --output_file_name combined_1_1 --do_train --do_eval --evaluate_during_training --logging_steps 10 --labels data/labels.txt --do_lower_case --overwrite_cache --overwrite_output_dir


# scores_plot.py

python3 scores_plot.py --file1 file_name_1.txt --file2 file_name_2.txt


# scores_plot_token.py

python3 scores_plot_token.py --file product_1_10_s.txt --labels "product-car,product-weapon"


# scores_plot_span.py

python3 scores_plot_span.py --file product_1_10_s.txt --labels "product-car,product-weapon"


# merge_and_convert_to_supervised.py

python3 merge_and_convert_to_supervised.py train_art_2_1.jsonl test_art_2_1.jsonl --output_dir


# create_episoded.py

python3 create_episodes.py --data_file data/inter/all.txt --labels_file data/labels.txt --N 2 --K 2 --num_episodes 20 --max_length 100 --output_file episodes_trial.jsonl