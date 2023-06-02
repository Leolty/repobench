from transformers import AutoTokenizer, AutoModel
from retriever.similarity import edit_similarity, jaccard_similarity, cosine_similarity
import torch
from model.unixcoder import UniXcoder


def retrieve(
    code,
    candidates: list,
    tokenizer = None,
    model = None,
    max_length: int = None,
    similarity: str = "jaccard"
    ):
    
    # check if the similarity is valid
    assert similarity in ["edit", "jaccard", "cosine"], "similarity must be one of edit, jaccard, cosine"

    if similarity == "cosine":
        assert model, "model must be provided if similarity is cosine"

        def get_embedding(code):
            if not code:
                # the type of model
                if isinstance(model, UniXcoder):
                    # this is for unixcoder
                    return torch.zeros(model.config.hidden_size).to(model.model.device)
                else:
                    return torch.zeros(model.config.hidden_size).to(model.device)

            if isinstance(model, UniXcoder):
                # this is for unixcoder
                tokens_ids = model.tokenize([code], max_length=max_length, mode="<encoder-only>")
                source_ids = torch.tensor(tokens_ids).to(model.model.device)
                with torch.no_grad():
                    _, code_embeddings = model(source_ids) # [1, 768]
                
                code_embeddings = torch.squeeze(code_embeddings) # [hidden_size]

            else:
                code_tokens=tokenizer.tokenize(code, max_length=max_length, truncation=True)
                tokens_ids=tokenizer.convert_tokens_to_ids(code_tokens)
                with torch.no_grad():
                    code_embeddings=model(torch.tensor(tokens_ids)[None,:].to(model.device))[0] # [1, seq_len, hidden_size]
                # calculate the mean of the embeddings
                code_embeddings = torch.mean(code_embeddings, dim=1) # [1, hidden_size]
                code_embeddings = torch.squeeze(code_embeddings) # [hidden_size]
            return code_embeddings
        
        code_embedding = get_embedding(code)
        candidates_embeddings = [ get_embedding(candidate) for candidate in candidates ]

        # calculate the cosine similarity between the code and the candidates
        sim_scores = []
        for i, candidate_embedding in enumerate(candidates_embeddings):
            sim_scores.append((i, cosine_similarity(code_embedding, candidate_embedding)))
    else:
        # candidates is a list of code strings
        # we need to sort the candidate index based on the edit similarity in a descending order
        sim_scores = []
        for i, candidate in enumerate(candidates):
            if similarity == "edit":
                sim_scores.append((i, edit_similarity(code, candidate, tokenizer)))
            elif similarity == "jaccard":
                sim_scores.append((i, jaccard_similarity(code, candidate, tokenizer)))
        
    
    # sort the candidate index based on the edit similarity in a descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # only return the index
    ranks = [ index for index, score in sim_scores ]

    return ranks