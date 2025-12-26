"""
Synthetic Data Generation Pipeline for IndicBERT
Combines 4 proven augmentation techniques for low-resource Indic languages
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from collections import Counter


# ===== STRATEGY 1: BACK-TRANSLATION =====
def backtranslation_augment ( texts, labels, target_count=2000 ) :
    """
    Use Google Translate API or mBart50 for back-translation
    Hindi -> English -> Hindi to create paraphrases
    """
    print( "\n🔄 Back-Translation Augmentation" )
    print( "NOTE: Requires Google Translate API or mBart50 model" )

    # Pseudo-code (implement with actual translation API)
    augmented_texts = []
    augmented_labels = []

    # Example using Google Translate (you need to install googletrans)
    try :
        from googletrans import Translator
        translator = Translator()

        for text, label in tqdm( zip( texts, labels ), total=len( texts ) ) :
            # Original
            augmented_texts.append( text )
            augmented_labels.append( label )

            # Translate to English and back
            try :
                eng = translator.translate( text, src='auto', dest='en' ).text
                back = translator.translate( eng, src='en', dest='hi' ).text
                augmented_texts.append( back )
                augmented_labels.append( label )
            except :
                continue

            if len( augmented_texts ) >= target_count :
                break

    except ImportError :
        print( "⚠️  Install googletrans: pip install googletrans==3.1.0a0" )
        print( "   Or use mBart50 for better quality" )
        return texts, labels

    return augmented_texts[:target_count], augmented_labels[:target_count]


# ===== STRATEGY 2: EASY DATA AUGMENTATION (EDA) =====
def eda_augment ( texts, labels, alpha=0.1, num_aug=1 ) :
    """
    Easy Data Augmentation (Wei & Zou 2019)
    - Random Deletion (RD)
    - Random Swap (RS)
    """
    print( "\n✂️  EDA: Random Deletion & Swap" )

    augmented_texts = []
    augmented_labels = []

    for text, label in tqdm( zip( texts, labels ), total=len( texts ) ) :
        # Original
        augmented_texts.append( text )
        augmented_labels.append( label )

        words = text.split()
        n_words = len( words )

        for _ in range( num_aug ) :
            # Random Deletion
            if random.random() < 0.5 and n_words > 3 :
                n_delete = max( 1, int( alpha * n_words ) )
                new_words = [w for w in words if random.random() > alpha]
                if len( new_words ) > 0 :
                    augmented_texts.append( ' '.join( new_words ) )
                    augmented_labels.append( label )

            # Random Swap
            else :
                new_words = words.copy()
                for _ in range( max( 1, int( alpha * n_words ) ) ) :
                    if len( new_words ) >= 2 :
                        idx1, idx2 = random.sample( range( len( new_words ) ), 2 )
                        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
                augmented_texts.append( ' '.join( new_words ) )
                augmented_labels.append( label )

    return augmented_texts, augmented_labels


# ===== STRATEGY 3: LLM-BASED GENERATION =====
def llm_generate_synthetic ( original_texts, labels, model_name="gpt-3.5-turbo",
                             samples_per_class=500 ) :
    """
    Use Claude API or GPT to generate synthetic Indic news samples
    """
    print( "\n🤖 LLM-Based Synthetic Generation" )
    print( "NOTE: Requires Anthropic/OpenAI API key" )

    # Example prompt template
    prompt_template = """Generate a realistic {language} news headline about {topic}.
The news should be classified as {label_type}.

Examples:
{examples}

Generate 1 new headline:"""

    # Pseudo-code (implement with actual API)
    synthetic_texts = []
    synthetic_labels = []

    # Group by label
    label_groups = {}
    for text, label in zip( original_texts, labels ) :
        if label not in label_groups :
            label_groups[label] = []
        label_groups[label].append( text )

    print( "⚠️  Implement this using Claude API or OpenAI API" )
    print( "   See: https://docs.anthropic.com/en/api/messages" )

    # For now, return empty (you need to implement API calls)
    return synthetic_texts, synthetic_labels


# ===== STRATEGY 4: CONTEXTUAL WORD SUBSTITUTION =====
def contextual_substitution ( texts, labels, num_aug=1 ) :
    """
    Replace words with contextually similar words
    For Indic languages, use IndicBERT's masked language model
    """
    print( "\n🔤 Contextual Word Substitution" )
    print( "NOTE: Requires IndicBERT MLM model" )

    try :
        from transformers import pipeline

        # Load masked language model
        fill_mask = pipeline(
            "fill-mask",
            model="ai4bharat/indic-bert",
            top_k=5
        )

        augmented_texts = []
        augmented_labels = []

        for text, label in tqdm( zip( texts, labels ), total=len( texts ) ) :
            # Original
            augmented_texts.append( text )
            augmented_labels.append( label )

            words = text.split()
            if len( words ) < 3 :
                continue

            for _ in range( num_aug ) :
                # Randomly mask 1-2 words
                new_words = words.copy()
                mask_indices = random.sample( range( len( words ) ),
                                              min( 2, len( words ) ) )

                for idx in mask_indices :
                    new_words[idx] = "[MASK]"

                masked_text = ' '.join( new_words )

                try :
                    # Get predictions
                    predictions = fill_mask( masked_text )

                    # Replace with top prediction
                    if isinstance( predictions, list ) and len( predictions ) > 0 :
                        filled_text = predictions[0]['sequence']
                        augmented_texts.append( filled_text )
                        augmented_labels.append( label )
                except :
                    continue

        return augmented_texts, augmented_labels

    except ImportError :
        print( "⚠️  Install transformers: pip install transformers" )
        return texts, labels


# ===== MAIN PIPELINE =====
def create_synthetic_dataset ( csv_path, output_path, target_size=10000 ) :
    """
    Main pipeline: Combines all augmentation strategies
    """
    print( "=" * 70 )
    print( "SYNTHETIC DATA GENERATION FOR INDICBERT" )
    print( "=" * 70 )

    # Load IndicBERT-routed data
    df = pd.read_csv( csv_path )
    df = df[df['route'] == 'indicbert'].copy()

    texts = df['text'].tolist()
    labels = df['label'].tolist()

    print( f"\nOriginal dataset: {len( texts )} samples" )
    print( f"Label distribution: {dict( Counter( labels ) )}" )

    # ===== STEP 1: Balance Original Data =====
    label_counts = Counter( labels )
    min_class = min( label_counts, key=label_counts.get )
    max_class = 1 - min_class

    # Oversample minority class
    min_indices = [i for i, l in enumerate( labels ) if l == min_class]
    max_indices = [i for i, l in enumerate( labels ) if l == max_class]

    # Duplicate minority class
    oversample_factor = label_counts[max_class] // label_counts[min_class]
    balanced_indices = max_indices + (min_indices * oversample_factor)

    balanced_texts = [texts[i] for i in balanced_indices]
    balanced_labels = [labels[i] for i in balanced_indices]

    print( f"\nAfter balancing: {len( balanced_texts )} samples" )
    print( f"Distribution: {dict( Counter( balanced_labels ) )}" )

    # ===== STEP 2: Apply Augmentation =====
    all_texts = balanced_texts.copy()
    all_labels = balanced_labels.copy()

    # Strategy 1: EDA (lightweight, always works)
    if len( all_texts ) < target_size * 0.5 :
        aug_texts, aug_labels = eda_augment(
            balanced_texts,
            balanced_labels,
            alpha=0.1,
            num_aug=2
        )
        all_texts.extend( aug_texts[len( balanced_texts ) :] )  # Add only new
        all_labels.extend( aug_labels[len( balanced_labels ) :] )
        print( f"After EDA: {len( all_texts )} samples" )

    # Strategy 2: Contextual Substitution
    if len( all_texts ) < target_size * 0.7 :
        aug_texts, aug_labels = contextual_substitution(
            balanced_texts[:1000],  # Limit to avoid slowdown
            balanced_labels[:1000],
            num_aug=1
        )
        all_texts.extend( aug_texts[1000 :] )
        all_labels.extend( aug_labels[1000 :] )
        print( f"After contextual sub: {len( all_texts )} samples" )

    # Strategy 3: Back-translation (optional, requires API)
    # Uncomment if you have translation API
    # if len(all_texts) < target_size:
    #     aug_texts, aug_labels = backtranslation_augment(
    #         balanced_texts[:500],
    #         balanced_labels[:500],
    #         target_count=target_size - len(all_texts)
    #     )
    #     all_texts.extend(aug_texts)
    #     all_labels.extend(aug_labels)

    # ===== STEP 3: Save Augmented Dataset =====
    # Shuffle
    combined = list( zip( all_texts, all_labels ) )
    random.shuffle( combined )
    all_texts, all_labels = zip( *combined )

    # Create DataFrame
    augmented_df = pd.DataFrame( {
        'text' : all_texts,
        'label' : all_labels,
        'source' : ['synthetic'] * len( all_texts )
    } )

    augmented_df.to_csv( output_path, index=False )

    print( "\n" + "=" * 70 )
    print( "AUGMENTATION COMPLETE" )
    print( "=" * 70 )
    print( f"✅ Saved: {output_path}" )
    print( f"Total samples: {len( augmented_df )}" )
    print( f"Final distribution: {dict( Counter( augmented_df['label'] ) )}" )

    return augmented_df


# ===== USAGE =====
if __name__ == "__main__" :
    # Configure paths
    INPUT_CSV = "C:/Users/somas/PycharmProjects/EXPLAIN/preprocessing/processed_data/combined_preprocessed.csv"
    OUTPUT_CSV = "C:/Users/somas/PycharmProjects/EXPLAIN/preprocessing/processed_data/indicbert_synthetic.csv"

    # Generate synthetic data
    synthetic_df = create_synthetic_dataset(
        csv_path=INPUT_CSV,
        output_path=OUTPUT_CSV,
        target_size=10000  # Target 10k samples
    )

    print( "\n📋 Next Steps:" )
    print( "1. Tokenize the synthetic dataset" )
    print( "2. Combine with original data (70% synthetic + 30% real)" )
    print( "3. Train IndicBERT with the augmented dataset" )
    print( "4. Compare performance against baseline" )