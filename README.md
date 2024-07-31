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


-------

## What combinations do we tested?

In this section I will like to describe more of our combinations (to organize and to prevent to do this same tests again).

<details>
<summary>Use of fixed LR || Using bigger LRs</summary>

- The max value of LR to the model converge is 0.00065, bigger than that the local min is to high. With this LR in some time the model will diverge, less than that help to fasten the convergence.

- Fixed LR: LR = 0.00.1 if applied in the worse subject of ADFEG help to decrease the f1-score. EVEN THOUGH the loss curve was perfectly decreasing! I think it overfitted the data.

</details>

<details>
<summary>The role of data augmentation</summary>

The custom data augmentation process was fixed in this code version!

- Increasing the variability and fixing previous errors in the project, help tp create more difficulty to the train the model, specially when the more variable data is not present;
- If we see the learning curves of LEFT_0 and LEFT_4 we understand that the overfitting is happening, due to the really good training loss, but the worse results in the inference.

</details>

<details>
<summary>Removing more layers from the encoder</summary>


</details>

<details>
<summary>Creating overlapped data</summary>


</details>

<details>
<summary>Using pre-trained model</summary>


</details>

<details>
<summary>Adding regularization techniques</summary>

- The dropout layers help the model to converge;
- Using L1 and L2 norms inside the concolution blocks returned worse results, 
possibly because of its magnitude in comparisson with the weights, idk, it didnt help at all.

</details>