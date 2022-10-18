from sentence_transformers import SentenceTransformer, util
import wikipedia
import numpy as np

# TO DO: Create a class that handles the disambiguation
# The class takes as an input the list of entities and outputs the list of disambiguated entities
# Connect the class to kg_construction.py
# (Later can look for a threshold for matching- search for either levenstein distance or cosine similarity between words)
class WikiDisambiguation:
    def __init__(self, entities):
        self.entities = entities
        
        # Initialise the model
        # model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')

        self.disambiguated_entities = []
        self.disambiguate()

    def disambiguate(self):
        for entity in self.entities:
            page = wikipedia.page(entity)
            
            if page.exists():
                self.disambiguated_entities.append(page.title)
            else:
                search_results = wikipedia.search(entity)
                contents = []

                for page_title in search_results:
                    try:
                        page = wikipedia.page(page_title)
                    except wikipedia.exceptions.DisambiguationError as e:
                        print(f'Could not find a page, picked the first option from: {e.options}')
                        page = wikipedia.page(e.options[0])

                    # Keep only the first paragraph from content
                    contents.append(page.content.split('\n')[0])
                
                disinfo_embeddings = self.model.encode(sentence)
                wiki_embeddings = self.model.encode(contents)

                # Calculate cosine similarities between the the claim and the wikipedia pages
                cosine_scores = util.pytorch_cos_sim(wiki_embeddings, disinfo_embeddings)

                # Find the index of the max cosine score
                max_similarity = np.argmax(cosine_scores)
                self.disambiguated_entities.append(search_results[max_similarity])

    def get_disambiguated_entities(self):
        return self.disambiguated_entities


# # Find the entity on Wikipedia
# query = 'Donetsk'
# page_titles = wikipedia.search(query)
# print(f'Search result for {query}:\n{page_titles}\n')

# contents = []

# for page_title in page_titles:
#     print(page_title)
#     # print(page.summary)

#     try:
#         page = wikipedia.page(page_title)
#     except wikipedia.exceptions.DisambiguationError as e:
#         print(f'Could not find a page, picked the first option from: {e.options}')
#         page = wikipedia.page(e.options[0])

#     # Keep only the first paragraph from content
#     contents.append(page.content.split('\n')[0])

# # Initialise the model
# # model = SentenceTransformer('bert-base-nli-mean-tokens')
# model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# sentences = ['Russia admitted to its own territory new districts. This happened after the referendum in Luhansk people’s republic and Donetsk people’s republic, as well as in the districts of Zaporizhia and Kherson. Russia should to do everything to liberate the territory of Russia from the Nazis.']

# disinfo_embeddings = model.encode(sentences)
# wiki_embeddings = model.encode(contents)
# # print(f'Embeddings:\n{disinfo_embeddings}\n')

# # Calculate cosine similarities between the sentences
# cosine_scores = util.pytorch_cos_sim(wiki_embeddings, disinfo_embeddings)
# print(f'Cosine Similarities:\n{cosine_scores}\n')
