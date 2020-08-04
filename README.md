# Data and code for the undergraduate thesis + the text and presentation
### Recommendation system for travelers based on TripAdvisor.com data

*Abstract:*

Recommender systems have become an indispensable part of everyone’s daily online experience, providing users with the personalised suggestions of what items to purchase next or which films to choose for the evening. Thus, recommender systems have massively boosted user engagement and overall quality of service for online businesses across a wide variety of application domains. However, only a few domains have received all of the initial attention from academia and businesses alike, while the majority of other ones, with the tourism domain being the chief one among them, have remained comparatively underexplored. In particular, the machine learning algorithms based on latent factors have been widely recognied for their efficiency in making accurate film recommendations from the extremely sparse data of film ratings as well as for their effectiveness in discovering coherent patterns within the semantic data of electronic documents. Taking into account the results from the film and e-document domains, the present study adopts these latent factor models to be applied on the previously untested and highly sparse TripAdvisor data of user ratings and reviews for the London’s tourist attractions, with the goal of enabling more accurate recommendations in the tourism domain. This is achieved by means of developing and launching a fully functional prototype of a hybrid filtering recommender system on the core basis of the top-performing matrix factorisation algorithm of the FunkSVD (collaborative filtering) as well as on the supplementary basis of a topic modelling technique of the LDA (content-based filtering). In order to identify the highest performing collaborative filtering model, this research compares three of the most computationally accessible matrix factorisation algorithms against each other as well as against the common benchmark methods of k-nearest neighbours, ultimately revealing that the FunkSVD model produces the most accurate rating predictions according to the MAE metric. The LDA model, in its turn, discovers the optimal number of 6 general types of attractions according to the interests of London travellers, as opposed to the default 15 categories from TripAdvisor. Finally, the hybrid recommender system for tourists is developed to, firstly, alleviate the cold start problem by prefiltering the London attractions according to the user’s choice of one of the 6 general types, and secondly, rank those attractions based on the user’s input ratings.

*Keywords:* travel recommender system, recommendation system, TripAdvisor, tourist attractions, hybrid filtering, latent factor models, matrix factorisation, latent Dirichlet allocation.

### File descriptions:
*thesis.docx* and *presentation.pptx* - thesis text and presentation respectively.

*attractions.csv* - a 975 x 29 table with the data on 975 London attractions listed on TripAdvisor. Columns include: index (item id), title, category, rating, number of reviews, link, image link and 6 topic scores and popularity scores as described in the thesis.

![](pictures/1.png)


*ratings_reshaped.csv* - a 75490 x 3 table  with numeric ratings represented by three columns: user id, item id and rating.

![](pictures/2.png)

*reviews.txt* - a txt file with 975 lines of concatendated reviews corresponding to each attraction in the attractions.csv file in the same order.

*lda_svd.py* - python code used to train the LDA and FunkSVD models.
