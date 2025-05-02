Unet with RGB, sampled points, hitmaps as imput channels (5 channels)
### Run training
- Add the dataset (including inputs, outputs and Positions).
- Run generate_samples.py to generate samples under different sparse sampling rate.
- Run los.py to generate losmap.
- Run hit_batch.py to generate hitmap.
- Modify dataset.py and train.py to accomadate your feature selection.
- Run train.py to start training.

*Note: Remember to update input channel number and feature dir when you need :)*
