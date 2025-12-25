import numpy as np
import torch


class TFIDFExplainer :
    """Wrapper for TF-IDF model to provide explanations"""

    def __init__ ( self, model, vectorizer ) :
        self.model = model
        self.vectorizer = vectorizer

    def explain ( self, text ) :
        """
        Explain TF-IDF model prediction
        Returns format consistent with transformer models
        """
        try :
            # Transform text
            X = self.vectorizer.transform( [text] )

            # Get prediction
            prediction = self.model.predict( X )[0]

            # Get probability/confidence
            if hasattr( self.model, 'predict_proba' ) :
                proba = self.model.predict_proba( X )[0]
                confidence = float( proba[prediction] )
            else :
                # For models without predict_proba, use decision function
                decision = self.model.decision_function( X )[0]
                confidence = float( 1 / (1 + np.exp( -decision )) )  # sigmoid

            # Get feature importance
            feature_names = self.vectorizer.get_feature_names_out()
            X_dense = X.toarray()[0]

            # Get top features by TF-IDF score
            non_zero_indices = np.where( X_dense > 0 )[0]

            if len( non_zero_indices ) == 0 :
                tokens_explanation = []
            else :
                # Get model coefficients if available
                if hasattr( self.model, 'coef_' ) :
                    coef = self.model.coef_[0] if len( self.model.coef_.shape ) == 2 else self.model.coef_
                    importance_scores = X_dense * coef
                else :
                    # Fallback to TF-IDF scores
                    importance_scores = X_dense

                # Create token explanations
                tokens_explanation = []
                for idx in non_zero_indices :
                    tokens_explanation.append( {
                        'token' : feature_names[idx],
                        'score' : float( importance_scores[idx] )
                    } )

                # Sort by absolute importance
                tokens_explanation.sort( key=lambda x : abs( x['score'] ), reverse=True )

                # Keep top 50 tokens
                tokens_explanation = tokens_explanation[:50]

            return {
                'prediction' : int( prediction ),
                'confidence' : confidence,
                'method' : 'TF-IDF',
                'tokens' : tokens_explanation
            }

        except Exception as e :
            return {
                'prediction' : 0,
                'confidence' : 0.0,
                'method' : 'TF-IDF',
                'tokens' : [],
                'error' : str( e )
            }