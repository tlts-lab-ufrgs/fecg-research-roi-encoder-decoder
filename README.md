# An end-to-end RoI-based encoder-decoder for fetal ECG recovery and QRS complex detection

When using this model, please cite the original paper: ``

## Removed ADFECG [17] r10 segments
Considering segment size of 512, the segment closed intervals are: [40-52], [56,60], 177, [179,190], [197,207], [311,320], [335, 341], 343, 365, 371, [393,412].

## Requirements
```tensorflow-2.14.0```
```keras-2.14.0```


## Code

- The proposed model is available at **models/ae_proposed**. To generate its weights, please run **main.py** after changing the local variables.
- To run a grid search over the hyperparameters, uncomment the loop segment in **loop_hyper.py** change the local variables and run the code. 
- In **data_load** dir you can find the subfunction that re-organizes the dataset data to the model understandable format
- To run evaluation on NI-FECG and NInFEA datasets, go to **model_eval/** dir
- Also, in **model_eval/** dir you can find MAE / MSE evalluation and peak detection evaluation - with the proposed method and Pan and Tomps Method. 
- The model weights are available at https://drive.google.com/drive/folders/1vZ9WOS__G-kFC5ivZKqLK6A9MWLHqgIu?usp=sharing

Don't forget to change the local variables! The main files are developed to understand ADFECG data format. 


-------

Any questions, you can send me an email to: juliacremus@gmail.com or julia.remus@inf.ufrgs.br
