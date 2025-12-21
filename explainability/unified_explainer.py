from explainability.integrated_gradients import IGExplainer


class UnifiedExplainer :
    def __init__ ( self, xlmr_model, xlmr_tokenizer,
                   indic_model, indic_tokenizer,
                   tfidf_explainer ) :

        self.xlmr_ig = IGExplainer( xlmr_model, xlmr_tokenizer )
        self.indic_ig = IGExplainer( indic_model, indic_tokenizer )
        self.tfidf = tfidf_explainer

    def explain ( self, text, model_name ) :
        if model_name == "xlmr" :
            return self.xlmr_ig.explain( text )
        elif model_name == "indicbert" :
            return self.indic_ig.explain( text )
        elif model_name == "tfidf" :
            return self.tfidf.explain( text )
        else :
            raise ValueError( "Unknown model" )
