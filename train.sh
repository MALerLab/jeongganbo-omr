# python3 train_omr.py test_setting.remove_borders=true data_path.train_aux=false +synth=vanilla
# python3 train_omr.py test_setting.remove_borders=true data_path.train_aux=false +synth=wnoise
# python3 train_omr.py test_setting.remove_borders=true data_path.train_aux=false +synth=full
# python3 train_omr.py test_setting.remove_borders=true +synth=vanilla
# python3 train_omr.py test_setting.remove_borders=true +synth=wnoise
# python3 train_omr.py test_setting.remove_borders=true +synth=full


python3 train_omr.py general.model_name=transformer_4M_with_aux_mix dataloader.mix_aux=true dataloader.batch_size=101 +synth=full
python3 train_omr.py general.model_name=transformer_4M_with_aux_freq_50 dataloader.aux_freq=50 +synth=full
python3 train_omr.py general.model_name=transformer_4M_with_aux_freq_100 dataloader.aux_freq=100 +synth=full
python3 train_omr.py general.model_name=transformer_4M_with_aux_freq_150 dataloader.aux_freq=150 +synth=full
python3 train_omr.py general.model_name=transformer_4M_with_aux_freq_200 dataloader.aux_freq=200 +synth=full