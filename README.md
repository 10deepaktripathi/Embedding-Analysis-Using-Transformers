
# Embedding Quality Evaluation for Health-related Data

This repository contains the analysis and evaluation of embedding quality for health-related data using various embedding models. The goal was to assess the effectiveness of different models in capturing the semantic meaning and context of health-related keywords and clinical notes.

# Directory Structure

- [Embedding Analysis1.ipynb](./Embedding%20Analysis1.ipynb): Notebook contains comparitive study across multiple pretrained and finetuned models.
- [model_tuning.ipynb](./model_tuning.ipynb): Notebook containing model architectures and parameters for fine-tuning clinical BERT model.
- [preprocess_data.ipynb](./preprocess_data.ipynb): Notebook containing code for preprocessing ClinicNotes and Medical Keyword files.
- [Detailed Report on Embedding Analysis](./detaild_report_on_embedding_analysis.pdf): A detailed PDF report about this case study
- [charts](./charts): Contains all the charts that were drawn on this project.
- [data](./data): Contains all the data files models are getting analyized or finetuned on.

Feel free to explore the notebooks and report for more information.


## Evaluation Approaches

Three main approaches were used to assess the embedding quality:

1. Cosine Similarity between Medical Keyword Pairs: The cosine similarity was calculated for pairs of medical keywords to measure their similarity. A higher cosine similarity score indicates greater similarity between the keywords.

2. Scatter Plot of Clinic Notes Embeddings: The clinic notes were transformed into embeddings using different models, and the embeddings were visualized using scatter plots. The goal was to observe if similar categories of clinic notes formed distinct clusters in the plot.
3. Perform K-Means clustering on clinic notes embeddings and then check which type of category's (gastroenterology, cardiovascular, neurology) data produces the similar type of embedding. In other ways we want to check which model is getting confused among different types of category’s data points.


## Embedding Models

Several embedding models were evaluated for their performance on health-related data:

1. Word2Vec: Word2Vec embeddings, obtained using pre-trained models like en_core_web_sm and en_core_web_lg from spaCy, were the initial baseline. However, they did not perform well in capturing health-related context.

2. ELMO: ELMO embeddings, which capture contextual information effectively, were used to generate embeddings for clinic notes. While ELMO performed better than Word2Vec, it was not specifically trained on health-related data.

3. BERT: BERT embeddings, including BERT base uncased, BioBERT, ClinicalBERT, and BlueBERT, were evaluated. </br>
   BERT Base uncased model excel at capturing context, but is not specifically trained on health-related data. BioBERT, ClinicalBERT, and BlueBERT are trained on health related data
   
  K-Means clustering result. Note that models is getting confused in putting datapoints in different clusters.
  
![Chart](KMeans_results/kmean_result.png)

K-Means clustering result after finetunning model with 3 different approaches.
1. Tune model to differentiate between right keyword pair and wrong keyword pair
2. Tune model on clinic notes to predict the notes category
3. Tune model first on clinic notes to predict right note's category and then further tune the same model to differentiate between right keyword pair and wrong keyword pair.

![Chart](KMeans_results/kmeans_on_tuned_models.png)

Scatter plots of clinic notes embeddings against clinic notes categories.
![Chart](charts/word2vec_96.png)
![Chart](charts/Elmo_1024.png)
![Chart](charts/BERT_768.png)
![Chart](charts/BioBERT_768.png)
![Chart](charts/ClinicalBERT_768.png)
![Chart](charts/BlueBERT_768.png)

</br>ClinicalBERT has captured the cluster better than any other model</br>

</br> Below is the cosine similarity result plot from each model.</br>

![Chart](charts/pretrained_model_comparion.png)

4. Fine-tuned ClinicalBERT: ClinicalBERT was further fine-tuned using clinic notes data to improve its performance on health-related embeddings. This model was trained on a 3-class classification task to predict the category given clinic notes.

![Chart](charts/keyword_pair_based_finetuned_model_768.png)
![Chart](charts/finetuned_model_on_clinical_notes_768.png)
![Chart](charts/Model_trained_on_clinical_notes_and_keyword_pair_both_768.png)


## Evaluation Results

The evaluation results revealed the following insights:

- Scatter plots: The scatter plots of clinic notes embeddings showed varying degrees of grouping. ClinicalBERT fine-tuned on clinic notes data exhibited the clearest groupings, indicating better capturing of health-related context.

- Keyword pair similarity: The models were also evaluated on the cosine similarity of medical keyword pairs. ClinicalBERT outperformed other models, achieving a similarity score of 0.8174 with finetuned and 0.8134 with pretrained.

## Conclusion

Based on the evaluation, all ClinicalBERT models, particularly when fine-tuned on clinic notes data and keyword pair both, showed the best performance in capturing health-related context. However, there is room for further research and improvement, including hyperparameter tuning and the incorporation of more health-specific training data.

For a detailed analysis and code implementation, please refer to the Jupyter Notebook and the respective model directories in this repository.
All models performence on keyword pair cosine similarity task

</br>Lastly, a consolidated performence of all models on cosine similarity task is summarized below</br>

![Chart](charts/tuned_model_comparion.png)
