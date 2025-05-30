Unet with Reflection, Tsum+fspl, log-scale distance, sampled point GT, hitmaps as input channels (5 channels)
### Run training
- Add the dataset (including inputs, outputs and Positions).
- Run generate_samples.py to generate samples under different sparse sampling rate.
- Run los.py to generate losmap.
- Run hit_batch.py to generate hitmap.
- Run Tsum_batch.py to generate Tsummap.
- Modify dataset.py and train.py to accomadate your feature selection.
- Run train.py to start training.

*Note: Remember to update input channel number and feature dir whenever you need :)*
